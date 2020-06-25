import os 
import sys 
import yaml 
import json 
import logging 
import termcolor as tc
from dict_deep import deep_get, deep_set, deep_del

import torch
import torch.nn as nn
import torch.nn.functional as F 
from ray.rllib.utils.annotations import override
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.torch.misc import normc_initializer, SlimFC
from ray.tune.logger import Logger, VALID_SUMMARY_TYPES
from ray.tune.util import flatten_dict
from ray.tune.result import (TRAINING_ITERATION, TIME_TOTAL_S, TIMESTEPS_TOTAL)

import logging
logger = logging.getLogger(__name__)


##########################################################################################
####################################     File Utils   ##################################
##########################################################################################


def mkdirs(*paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


def eval_token(token):
    """ convert string token to int, float or str """ 
    if token.isnumeric():
        return int(token)
    try:
        return float(token)
    except:
        return token

def read_file(file_path, sep=","):
    """ read a file (json, yaml, csv, txt)
    """
    if len(file_path) < 1 or not os.path.exists(file_path):
        return None 
    # load file 
    f = open(file_path, "r")
    if "json" in file_path:
        data = json.load(f)
    elif "yaml" in file_path:
        data = yaml.load(f)
    else:
        sep = sep if "csv" in file_path else " "
        data = []
        for line in f.readlines():
            line_post = [eval_token(t) for t in line.strip().split(sep)]
            # if only sinlge item in line 
            if len(line_post) == 1:
                line_post = line_post[0]
            if len(line_post) > 0:
                data.append(line_post)
    f.close()
    return data


def overwrite_config(line, config):
    """ args.overwrite has a string to overwrite configs 
    overwrite string format (flat hierarchy): 'nested_name-nested_type-value; ...' 
    nested_name: a.b.c; nested_type: type or list.type 
    """
    if line is None or len(line) == 0:
        return config 
    # parse overwrite string
    for item in line:
        cname, ctype, cval = [e.strip() for e in item.strip().split("-")]
        types = ctype.split(".")
        base_type = types[-1]
        # make value 
        if len(types) > 1:  # list of basic types 
            val = cval[1:-1].split(",")    # exclude brackets []
            if base_type == "bool":
                val = [False if e.strip() == "false" else True for e in val] 
            else:
                val = [eval(base_type)(e.strip()) for e in val]
            val = eval(types[0])(val)   # only 2 levels allowed
        else:   # int, float, bool, str
            if base_type == "bool": 
                val = False if cval == "false" else True
            else:
                val = eval(base_type)(cval)
        # update config 
        deep_set(config, cname, val)
    return config 
            


##########################################################################################
####################################     Model Utils   ##################################
##########################################################################################


# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11
def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15
def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


##########################################  Distributions  

class TorchRelaxedOneHotCategorical(TorchDistributionWrapper):
    """Wrapper class for PyTorch RelaxedOneHotCategorical distribution."""

    @override(ActionDistribution)
    def __init__(self, inputs, model, temperature=1.0):
        self.dist = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(temperature=temperature, logits=inputs)

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(action_space, model_config):
        return action_space.n

    def rsample(self):
        return self.dist.rsample()


class TorchNormal(TorchDistributionWrapper):
    """Wrapper for PyTorch MultivariateNormal distribution."""
    @override(ActionDistribution)
    def __init__(self, inputs, model, covar=1e-3):
        dim = inputs.shape[-1]
        self.dist = torch.distributions.multivariate_normal.MultivariateNormal(
                        inputs, torch.eye(dim)*covar)

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(action_space, model_config):
        # return action_space.n
        return action_space.shape[0]

    def rsample(self):
        return self.dist.rsample()


##########################################  MLP policy 

class MyFullyConnectedNetwork(nn.Module):
    """Generic fully connected network."""

    def __init__(self, num_inputs, num_outputs, model_config):
        nn.Module.__init__(self)

        hiddens = model_config["hiddens"]
        activation = getattr(nn, model_config["activation"])

        logger.debug("Constructing fcnet {} {}".format(hiddens, activation))
        layers = []
        last_layer_size = num_inputs
        for size in hiddens:
            layers.append(
                SlimFC(
                    in_size=last_layer_size,
                    out_size=size,
                    #initializer=normc_initializer(1.0),
                    activation_fn=activation))
            last_layer_size = size

        self._hidden_layers = nn.Sequential(*layers)

        self._logits = SlimFC(
            in_size=last_layer_size,
            out_size=num_outputs,
            #initializer=normc_initializer(0.01),
            activation_fn=None)

    def forward(self, input, hidden_state=None, seq_lens=None):
        features = self._hidden_layers(input)
        logits = self._logits(features)
        return logits, hidden_state


##########################################  RNN policy 

class MyRecurrentNetwork(nn.Module):
    """Generic rnn network."""

    def __init__(self, num_inputs, num_outputs, model_config):
        nn.Module.__init__(self)

        # Input embedding layer 
        self.embed_dim = model_config["embed_size"]
        self.embedding = SlimFC(
            in_size=num_inputs,
            out_size=self.embed_dim,
            #initializer=normc_initializer(0.01),
            activation_fn=model_config["embed_act"]
        )

        # RNN 
        rnn_hidden_size = model_config["rnn_hidden_size"]
        self.rnn = nn.GRUCell(self.embed_dim, rnn_hidden_size)
        logger.debug("Constructing RNN {} {}".format(rnn_hidden_size, activation))

        # output 
        self._logits = SlimFC(
            in_size=rnn_hidden_size,
            out_size=num_outputs,
            #initializer=normc_initializer(0.01),
            activation_fn=None)

    def forward(self, input, hidden_state=None, seq_lens=None):
        features = self.embedding(input)
        hx = self.rnn(features, hidden_state)
        logits = self._logits(hx)
        return logits, hx


##########################################  GNN policy 

class MyGraphNetwork(nn.Module):
    """ Generic graph net. 
    reference: https://github.com/lrjconan/LanczosNetwork --> GAT model
    """
    def __init__(self, num_inputs, num_outputs, model_config,  num_edgetype=1):
        nn.Module.__init__(self)

        # Input embedding layer 
        self.embed_dim = model_config["embed_size"]
        self.embedding = SlimFC(
            in_size=num_inputs,
            out_size=self.embed_dim,
            #initializer=normc_initializer(0.01),
            activation_fn=model_config["embed_act"]
        )

        # Propagatoin layers 
        hidden_dims = model_config["hiddens"]
        dim_list = [embed_dim] + hidden_dims + [num_outputs]
        self.num_layer = len(dim_list) - 2

        self.num_edgetype = num_edgetype
        self.num_heads = model_config.get("num_heads", [1]*self.num_layer)
        self.dropout = model_config.get("dropout", 0.0)
        logger.debug("Constructing GNN: dims-{} heads:{}".format(hidden_dims, self.num_heads))

        self.filter = nn.ModuleList([
            nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(
                        dim_list[tt] *
                        (int(tt == 0) + int(tt != 0) * self.num_heads[tt] * self.num_edgetype),
                        dim_list[tt + 1], 
                        bias=False
                    ) for _ in range(self.num_heads[tt])     # heads 
                ]) for _ in range(self.num_edgetype)       # edge types
            ]) for tt in range(self.num_layer)      # layers
        ])      # 1st layer output concat all heads, so for first layer, input is not concat

        # Attention layers 
        self.att_net_1 = nn.ModuleList([
            nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(dim_list[tt + 1], 1)
                    for _ in range(self.num_heads[tt])
                ]) for _ in range(self.num_edgetype)
            ]) for tt in range(self.num_layer)
        ])

        self.att_net_2 = nn.ModuleList([
            nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(dim_list[tt + 1], 1)
                    for _ in range(self.num_heads[tt])
                ]) for _ in range(self.num_edgetype)
            ]) for tt in range(self.num_layer)
        ])

        # Biases
        self.state_bias = [
            [[None] * self.num_heads[tt]] * self.num_edgetype 
            for tt in range(self.num_layer)
        ]
        for tt in range(self.num_layer):
            for jj in range(self.num_edgetype):
                for ii in range(self.num_heads[tt]):
                    self.state_bias[tt][jj][ii] = torch.nn.Parameter(
                                            torch.zeros(dim_list[tt + 1]))
                    self.register_parameter('bias_{}_{}_{}'.format(ii, jj, tt),
                                            self.state_bias[tt][jj][ii])

        # output layer (could extend to MLP with distribution heads)
        self.output_func = nn.Sequential(*[nn.Linear(dim_list[-2], dim_list[-1])])
        self.output_level = model_config["output_level"]    # graph or node


    def forward(self, nodes, edges, masks=None):
        """
        Args:
            - nodes: (B, N, I)
            - edges: (B, N, N, E)
            - masks: (B, N)
        Returns:
            - out: (B, N, O) or (B, O) depending on `output_level`
        """
        batch_size, num_node, _ = node_feat.shape
        state = self.embedding(nodes)  # (B, N, D)

        for tt in range(self.num_layer):
            h = []

            # transform & aggregate features
            for jj in range(self.num_edgetype):
                for ii in range(self.num_heads[tt]):
                    
                    # transformed features
                    state_head = F.dropout(state, self.dropout, training=self.training)
                    Wh = self.filter[tt][jj][ii](
                        state_head.view(batch_size * num_node, -1)
                    ).view(batch_size, num_node, -1)  # (B, N, D)

                    # attention weights
                    att_weights_1 = self.att_net_1[tt][jj][ii](Wh)  # (B, N, 1)
                    att_weights_2 = self.att_net_2[tt][jj][ii](Wh)  # (B, N, 1)
                    att_weights = att_weights_1 + att_weights_2.transpose(1, 2)  # (B, N, N) dense matrix
                    att_weights = F.softmax(
                        F.leaky_relu(att_weights, negative_slope=0.2) + edges[:, :, :, jj],
                        dim=1)
                    
                    # dropout attn weights and features
                    att_weights = F.dropout(
                        att_weights, self.dropout, training=self.training)  # (B, N, N)
                    Wh = F.dropout(Wh, self.dropout, training=self.training)  # (B, N, D)

                    # aggregation step
                    msg = torch.bmm(att_weights, Wh) + self.state_bias[tt][jj][ii].view(1, 1, -1)
                    if tt == self.num_layer - 1:
                        msg = F.elu(msg)
                    h += [msg]  # (B, N, D)

            # propagation step 
            if tt == self.num_layer - 1:
                state = torch.mean(torch.stack(h, dim=0), dim=0)  # (B, N, D), average all heads & edges
            else:
                state = torch.cat(h, dim=2)     # (B, N, D * #edge_types * #heads)

        # output
        out = self.output_func(
            state.view(batch_size * num_node, -1)
        ).view(batch_size, num_node, -1)    # (B, N, O)
        # if output is `graph-level`, out is now (B, N, 1), convert to (B, 1)
        if self.output_level == "graph":
            if masks is not None:
                out = out.squeeze() * masks   # (B, N)
            out = torch.mean(out, dim=1)    # simple sum, could extend to weighted sum (attention)

        return out 



def construct_graph_input(obs_n, state, **kwargs):
    """ 
    Arguments:
        - obs_n: observation object returned from env 
                assume {"states": states, "masks": masks}
        - state: prev recurrent state 
    Returns:
        - nodes, edges, masks, state
    """
    nodes = torch.as_tensor(obs_n["states"])
    n = nodes.shape[0]
    edges = torch.ones(n, n)
    masks = torch.as_tensor(obs_n["masks"])





##########################################################################################
####################################     Logger   ##################################
##########################################################################################


# from https://github.com/ray-project/ray/blob/master/python/ray/tune/logger.py
class TBXLogger(Logger):
    """TensorBoard Logger.
    Automatically flattens nested dicts to show on TensorBoard:
        {"a": {"b": 1, "c": 2}} -> {"a/b": 1, "a/c": 2}
    """

    def _init(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            logger.error("Upgrade to the latest pytorch.")
            raise
        self._file_writer = SummaryWriter(self.logdir, flush_secs=10)
        self.last_result = None

    def on_result(self, result):
        
        step = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]
        tmp = result.copy()
        for k in [
                "config", "pid", "timestamp", TIME_TOTAL_S, TRAINING_ITERATION
        ]:
            if k in tmp:
                del tmp[k]  # not useful to log these

        flat_result = flatten_dict(tmp, delimiter="/")
        path = ["ray", "tune"]
        valid_result = {
            "/".join(path + [attr]): value
            for attr, value in flat_result.items()
            if type(value) in VALID_SUMMARY_TYPES
        }

        for attr, value in valid_result.items():
            self._file_writer.add_scalar(attr, value, global_step=step)
        self.last_result = valid_result
        self._file_writer.flush()

    def flush(self):
        if self._file_writer is not None:
            self._file_writer.flush()

    def close(self):
        if self._file_writer is not None:
            self._file_writer.close()



def build_TBXLogger(*args):
    """ factory function for building tensorboard logger with filtered loggings 
    """
    all_fields = [
        'episode_reward_max', 
        'episode_reward_min', 
        'episode_reward_mean', 
        'episode_len_mean', 
        'episodes_this_iter', 
        'policy_reward_min', 
        'policy_reward_max', 
        'policy_reward_mean', 
        'custom_metrics', 
        'sampler_perf', 
        'off_policy_estimator', 
        'info', 
        'timesteps_this_iter', 
        'done', 
        'timesteps_total', 
        'episodes_total', 
        'training_iteration', 
        'experiment_id', 
        'date', 
        'timestamp', 
        'time_this_iter_s', 
        'time_total_s', 
        'pid', 
        'hostname', 
        'node_ip', 
        'config', 
        'time_since_restore', 
        'timesteps_since_restore', 
        'iterations_since_restore'
    ]

    class FilteredTBXLogger(TBXLogger):
        """ modifications  
        - with modified `on_result` method to log out only required fields 
        - with `on_result` also logs rollout frames 
        """
        keep_fields = args
        log_videos = True 
        fps = 4 
        log_sys_usage = True 

        def on_result(self, result):
            step = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]
            tmp = result.copy()
            for k in [
                    "config", "pid", "timestamp", TIME_TOTAL_S, TRAINING_ITERATION
            ]:
                if k in tmp:
                    del tmp[k]  # not useful to log these

            # log system usage
            perf = result.get("perf", None)
            if FilteredTBXLogger.log_sys_usage and perf is not None:
                self.log_system_usage(step, perf)

            flat_result = flatten_dict(tmp, delimiter="/")
            path = ["scalars"]
            valid_result = {
                "/".join(path + [attr]): value
                for attr, value in flat_result.items()
                if type(value) in VALID_SUMMARY_TYPES 
                and attr in FilteredTBXLogger.keep_fields
            }

            # log scalars 
            for attr, value in valid_result.items():
                self._file_writer.add_scalar(attr, value, global_step=step)

            # log videos 
            videos = result.get("eval_frames", [])
            if FilteredTBXLogger.log_videos and len(videos) > 0:
                self.log_videos(step, videos, "rollout_frames")

            self.last_result = valid_result
            self._file_writer.flush()

        def log_images(self, step, images, tag):
            """ show images on tb, e.g. sequential graph structures, images: (N,C,H,W) """ 
            self._file_writer.add_images(tag, images, global_step=step)

        def log_videos(self, step, videos, tag):
            """ show rollouts on tb, videos: (T,H,W,C) """
            t, h, w, c = videos.shape
            # tb accepts (N,T,C,H,W)
            vid_tensor = torch.as_tensor(videos).permute(0,3,1,2).reshape(-1,t,c,h,w)   
            self._file_writer.add_video(tag, vid_tensor, global_step=step, fps=FilteredTBXLogger.fps)

        def log_system_usage(self, step, perf):
            """ cpu, gpu, ram usage """
            for n, v in perf.items():
                self._file_writer.add_scalar("sys/"+n, v, global_step=step)

    return FilteredTBXLogger



class CustomStdOut(object):
    """ self refers to trainer """
    def _log_result(self, result):
        log_interval = self.config.get("std_log_interval", 50)
        if result["training_iteration"] % log_interval == 0:
            try:
                print("iter: {}, steps: {}, episodes: {}, mean epi reward: {:.4f}, agent epi reward: {}, time: {}".format(
                    result["training_iteration"],
                    result["timesteps_total"],
                    result["episodes_total"],
                    result["episode_reward_mean"],
                    {k: round(v, 4) for k, v in result["policy_reward_mean"].items()},
                    round(result["time_total_s"] - self.cur_time, 3),
                ))
            except:
                pass
            self.cur_time = result["time_total_s"]



class ColoredStdOut(object):
    """ colored text, use termcolor, reference: https://pypi.org/project/termcolor/
        grey, red, green, yellow, blue, magenta, cyan, white
    """
    def _log_result(self, result):
        colors = ["yellow", "red", "blue"]
        log_interval = self.config.get("std_log_interval", 50)

        if result["training_iteration"] % log_interval == 0:
            try:
                # log system usage 
                cpu_percent = result["perf"]["cpu_util_percent"] / 100 if "perf" in result and "cpu_util_percent" in result["perf"] else "-"
                ram_percent = result["perf"]["ram_util_percent"] / 100 if "perf" in result and "ram_util_percent" in result["perf"] else "-"

                print("iter: {}, steps: {}, episodes: {}, time: {}, cpu: {}, ram: {}".format(
                    tc.colored(result["training_iteration"], colors[0]),
                    tc.colored(result["timesteps_total"], colors[0]),
                    tc.colored(result["episodes_total"], colors[0]),
                    tc.colored(round(result["time_total_s"] - self.cur_time, 3), colors[1]),
                    tc.colored(round(cpu_percent, 3), colors[1]),
                    tc.colored(round(ram_percent, 3), colors[1]),
                ))
                print("\t mean epi reward: {}, agent epi reward: {}".format(
                    tc.colored(round(result["episode_reward_mean"], 4), colors[2]),
                    tc.colored({k: round(v, 4) for k, v in result["policy_reward_mean"].items()}, colors[2]),
                ))

            except:
                pass
            self.cur_time = result["time_total_s"]


class Std2FileLogger(Logger):
    """ manage logging to sys.std
    """
    def _init(self):
        logger = logging.getLogger("maddpg")
        fileHandler = logging.FileHandler(os.path.join(self.logdir, "log.txt"))
        logger.addHandler(fileHandler)
        self.logger = logger 

    def on_result(self, result):
        
        step = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]
        tmp = result.copy()
        for k in [
                "config", "pid", "timestamp", TIME_TOTAL_S, TRAINING_ITERATION
        ]:
            if k in tmp:
                del tmp[k]  # not useful to log these

        flat_result = flatten_dict(tmp, delimiter="/")
        path = ["ray", "tune"]
        valid_result = {
            "/".join(path + [attr]): value
            for attr, value in flat_result.items()
            if type(value) in VALID_SUMMARY_TYPES
        }

        for attr, value in valid_result.items():
            self._file_writer.add_scalar(attr, value, global_step=step)
        self.last_result = valid_result
        self._file_writer.flush()

    def flush(self):
        pass

    def close(self):
        pass





##########################################################################################
####################################     Tests   ##################################
##########################################################################################


if __name__ == "__main__":
    # test overwrite config 
    test_string = "a.b int 3; dd.d.e list.bool [false,true,true]" 
    config = {
        "a": {"b": 0}, "c": 1.2, "dd": {"d": {"e": [1,2,3]}}
    }
    config = overwrite_config(test_string, config )
    # import pdb; pdb.set_trace()
    print("finished...")