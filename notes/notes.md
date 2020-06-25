# Notes 

- register_trainer 
- register_env 

- ray.tune.run(experiment)
- ray.tune.run_experiments(experiments)
- each experiment specified as Experiment object, can be instantiated from json dict 

- Tensorboard logger from torch, update to latest torch, tensorflow and tensorboard

<!---
##############################################################################
##############################################################################
##############################################################################
-->

## Trainer

Example Agent, reference: https://github.com/ray-project/ray/blob/5d7afe8092f5521a8faa8cdfc916dcd9a5848023/rllib/contrib/random_agent/random_agent.py

- Trainable 
    - __init__ calls _setup, _setup in trainer, also set up logger 
    - _setup calls _init, _init in trainer template --> build_trainer --> trainer cls template
    - _init finds policy class, instantiates training workers (takes in env_creator, policy, config), optimizer (takes in workers and optimizer configs)
    - trainer.train
        - reference: https://github.com/ray-project/ray/blob/d51583dbd6dc9c082764b9ec06349678aaa71078/python/ray/tune/trainable.py
        - self._train, override by trainer_template --> build_trainer
        - update total_timesteps, total_episodes and training iterations in trainable.train
        - call self._log_result at the end, takes in result and call, default is 
        ```python 
        self._result_logger.on_result(result)
        ```
        but will be override by trainer or mixins 


- GenericOffPolicyTrainer
    - references:
        - GenericOffPolicyTrainer: https://github.com/ray-project/ray/blob/master/rllib/agents/dqn/dqn.py 
        - build_trainer: https://github.com/ray-project/ray/blob/d51583dbd6dc9c082764b9ec06349678aaa71078/rllib/agents/trainer_template.py#L17
    - upon `_setup`, trainer takes in experiment config, merge with default_config
    - to allow merging with new config keys, set 
    ```python
    Trainer._allow_unknown_configs = True
    Trainer._allow_unknown_subkeys = [
        "tf_session_args", "local_tf_session_args", "env_config", "model",
        "optimizer", "multiagent", "custom_resources_per_worker",
        "evaluation_config"
    ]
    ```
    reference: https://github.com/ray-project/ray/blob/d51583dbd6dc9c082764b9ec06349678aaa71078/rllib/agents/trainer.py#L488
    - trainer.collect_metrics
        - call self.optimizer.collect_metrics
    - trainer._evaluate
        - call _before_evaluate, evaluation_workers restore, iterative evaluation_workers sample, collect_metrics(self.evaluation_workers.local_worker()) sequentially 
        - return {"evaluation": metrics}
    - trainer._make_workers
        - takes in env_creator, policy, config, num_workers
        - calls
        ```python
        WorkerSet(
            env_creator,
            policy,
            config,
            num_workers=num_workers,
            logdir=self.logdir)
        ```
    - trainer.compute_actions
        - takes in observation, state=None, prev_action=None, prev_reward=None, info=None,policy_id=DEFAULT_POLICY_ID, full_fetch=False
        - support rnn policy too 
        - get policy from policy id (policy set already stored in trainer)
        - either return computed action or full output from policy (computed action, state, other info, etc)
    - trainer._log_results 
        - call `on_train_result` (takes in trainer and result) from config, then call trainable._log_result (call `on_result` on list of loggers in trainable._result_logger)
    - trainer.train
        - call Trainable.train 


- Trainer template --> build_trainer
    - full init arguments
    ```python
        name,
        default_policy,
        default_config=None,
        validate_config=None,
        get_initial_state=None,
        get_policy_class=None,
        before_init=None,
        make_workers=None,
        make_policy_optimizer=None,
        after_init=None,
        before_train_step=None,
        after_optimizer_step=None,
        after_train_result=None,
        collect_metrics_fn=None,
        before_evaluate_fn=None,
        mixins=None
    ```
    - reference: https://github.com/ray-project/ray/blob/d51583dbd6dc9c082764b9ec06349678aaa71078/rllib/agents/trainer_template.py#L17
    - init with policy, workers, optimizer, 
    - ray/trainer_template.py --> _train does the actual training 
        - call before_train_step, iterative optimizer.step, collect_metrics, after_train_result sequentially 
        - reference: https://github.com/ray-project/ray/blob/d51583dbd6dc9c082764b9ec06349678aaa71078/rllib/agents/trainer_template.py#L17 
    - with_updates: Build a copy of this trainer with the specified overrides.
    - mixins, use add_mixins method to contruct base class for trainer, such that it inherits all classes in mixins, later mixins takes higer priority
    reference: https://github.com/ray-project/ray/blob/d51583dbd6dc9c082764b9ec06349678aaa71078/rllib/utils/__init__.py


<!---
##############################################################################
##############################################################################
##############################################################################
-->

## Evaluation
"""
to add new summary to metrics logging, 
refer to ray/rllib/evaluation/metrics.py  --> summarize_episodes
"""

rollout_metrics.py
- reference: https://github.com/ray-project/ray/blob/d51583dbd6dc9c082764b9ec06349678aaa71078/rllib/evaluation/rollout_metrics.py
```python
RolloutMetrics = collections.namedtuple("RolloutMetrics", [
    "episode_length", "episode_reward", "agent_rewards", "custom_metrics",
    "perf_stats"
])
```


metrics.py
- reference: https://github.com/ray-project/ray/blob/d51583dbd6dc9c082764b9ec06349678aaa71078/rllib/evaluation/metrics.py
- collect_metrics
    - gathers episode metrics from RolloutWorker instances
    - calls `collect_episodes` and `summarize_episodes` sequentially 
- collect_episodes
    - takes in `local_worker`, `remote_workers`, `to_be_collected`, `timeout_seconds`
    - return `episodes`, `to_be_collected`
- summarize_episodes
    - takes in 
        - episodes: smoothed set of episodes including historical ones
        - new_episodes: just the new episodes in this iteration
    - return dict of metrics 
    ```python
    dict(
        episode_reward_max=max_reward,
        episode_reward_min=min_reward,
        episode_reward_mean=avg_reward,
        episode_len_mean=avg_length,
        episodes_this_iter=len(new_episodes),
        policy_reward_min=policy_reward_min,
        policy_reward_max=policy_reward_max,
        policy_reward_mean=policy_reward_mean,
        custom_metrics=dict(custom_metrics),
        sampler_perf=dict(perf_stats),
        off_policy_estimator=dict(estimators))
    ```


MultiAgentEpisode
- reference: https://github.com/ray-project/ray/blob/d51583dbd6dc9c082764b9ec06349678aaa71078/rllib/evaluation/episode.py
- tracks the current state of a (possibly multi-agent) episode


rollout.py
- reference: https://github.com/ray-project/ray/blob/d51583dbd6dc9c082764b9ec06349678aaa71078/rllib/optimizers/rollout.py
- collect_samples
    - takes in `agents`, `sample_batch_size`, `num_envs_per_worker`, `train_batch_size`
    - collects at least train_batch_size samples, never discarding any
    - returns SampleBatch 
    - launch sampling tasks with 
    ```python
    for agent in agents:
        fut_sample = agent.sample.remote()
        agent_dict[fut_sample] = agent
    ```
    - # Only launch more tasks if we don't already have enough pending
    ```python
    pending = len(agent_dict) * sample_batch_size * num_envs_per_worker
    if num_timesteps_so_far + pending < train_batch_size:
        fut_sample2 = agent.sample.remote()
        agent_dict[fut_sample2] = agent
    ```


ReplayBuffer
- reference: https://github.com/ray-project/ray/blob/d51583dbd6dc9c082764b9ec06349678aaa71078/rllib/optimizers/replay_buffer.py
- takes in `size`, Max number of transitions to store in the buffer. When the buffer overflows the old memories are dropped.
- add 
    - takes in `obs_t`, `action`, `reward`, `obs_tp1`, `done`, `weight`
    - push `data = (obs_t, action, reward, obs_tp1, done)` to buffer 
- _encode_sample
    - turn data tuples to tuple of data columns 
    - return `(np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones))`
- sample 
    - takes in `batch_size`
    - sample indices then encode_sample


PrioritizedReplayBuffer
- additionally takes in `alpha` (float) how much prioritization is used (0 - no prioritization, 1 - full prioritization)

<!---
##############################################################################
##############################################################################
##############################################################################
-->

## Workers 

EvaluatorInterface
- reference: https://github.com/ray-project/ray/blob/d51583dbd6dc9c082764b9ec06349678aaa71078/rllib/evaluation/interface.py
- the interface between policy optimizers and policy evaluation
- sample
    - returns SampleBatch|MultiAgentBatch: A columnar batch of experiences
- learn_on_batch
    - equivalent to `apply_gradients(compute_gradients(samples))`


RolloutWorker
- reference: https://github.com/ray-project/ray/blob/d51583dbd6dc9c082764b9ec06349678aaa71078/rllib/evaluation/rollout_worker.py
- subclass **EvaluatorInterface**
- this class wraps a policy instance and an environment class to collect experiences from the environment. You can create many replicas of this class as Ray actors to scale RL training
- important arguments:
    - env_creator (func): Function that returns a gym.Env given an EnvContext wrapped configuration.
    - policy (class|dict): Either a class implementing Policy, or a dictionary of policy id strings to (Policy, obs_space, action_space, config) tuples. If a dict is specified, then we are in multi-agent mode and a policy_mapping_fn should also be set.
    - policy_mapping_fn (func): A function that maps agent ids to policy ids in multi-agent mode. This function will be called each time a new agent appears in an episode, to bind that agent to a policy for the duration of the episode.
    - batch_steps (int): The target number of env transitions to include in each sample batch returned from this worker.
    - atch_mode (str): One of the following batch modes: "truncate_episodes", "complete_episodes"    
    - episode_horizon (int): Whether to stop episodes at this horizon.
    - num_envs (int): If more than one, will create multiple envs and vectorize the computation of actions. This has no effect if the env already implements VectorEnv.
    - env_config (dict): Config to pass to the env creator.
    - model_config (dict): Config to use when creating the policy model.
    - policy_config (dict): Config to pass to the policy. In the multi-agent case, this config will be merged with the per-policy configs specified by `policy`.
    - worker_index (int): For remote workers, this should be set to a non-zero and unique value. This index is passed to created envs through EnvContext so that envs can be configured per worker.
    - monitor_path (str): Write out episode stats and videos to this directory if specified.
    - log_dir (str): Directory where logs can be placed.
    - seed (int): Set the seed of both np and tf to this value to to ensure each remote worker has unique exploration behavior.
- important attributes
    - `self.async_env`, BaseEnv
    - `self.sampler`, AsyncSampler or SyncSampler
    - `self.reward_estimators`, ImportanceSamplingEstimator, WeightedImportanceSamplingEstimator
    - `self.input_reader`
    - `self.output_writer`
- sample 
    - evaluate the current policies and return a batch of experiences
    - return SampleBatch|MultiAgentBatch
    - keep calling `self.input_reader.next()` until batch size is met 
- get_metrics
    - get metrics from `self.sampler` and each in `self.reward_estimators`


WorkerSet 
- reference: https://github.com/ray-project/ray/blob/d51583dbd6dc9c082764b9ec06349678aaa71078/rllib/evaluation/worker_set.py
- there must be one local worker copy, and zero or more remote workers
- takes in `env_creator`, `policy`, `trainer_config`, `num_workers`, `logdir`
- important attributes
    - `self._local_worker`
    - `self._remote_workers` (list)
- add_workers 
    - takes in `num_workers`
    ```python
    cls = RolloutWorker.as_remote(**remote_args).remote
    self._remote_workers.extend([
        self._make_worker(cls, self._env_creator, self._policy, i + 1,
                            self._remote_config) for i in range(num_workers)
        ])
    ```
- _make_worker
    - takes in `cls`, `env_creator`, `policy`, `worker_index`, `config`
    - call `cls` (RolloutWorker) with arguments `env_creator`, `policy`, etc


<!---
##############################################################################
##############################################################################
##############################################################################
-->

## Optimizers 

PolicyOptimizer
- reference: https://github.com/ray-project/ray/blob/d51583dbd6dc9c082764b9ec06349678aaa71078/rllib/optimizers/policy_optimizer.py
- important attributes 
    - config (dict): The JSON configuration passed to this optimizer.
    - workers (WorkerSet): The set of rollout workers to use.
    - num_steps_trained (int): Number of timesteps trained on so far.
    - num_steps_sampled (int): Number of timesteps sampled so far.
- step 
- collect_metrics
    - returns worker and optimizer stats
    - returns a training result dict from worker metrics with `info` replaced with stats from self
    - calls `collect_episodes`, `summarize_episodes` to get result dict 
- stats, return dict of `num_steps_trained` and `num_steps_sampled`
- save, restore
- reset, foreach_worker, foreach_worker_with_index: methods related to `self.workers`


SyncSamplesOptimizer
- reference: https://github.com/ray-project/ray/blob/d51583dbd6dc9c082764b9ec06349678aaa71078/rllib/optimizers/sync_samples_optimizer.py
- subclass `PolicyOptimizer`
- in each step, this optimizer pulls samples from a number of remote workers, concatenates them, and then updates a local model. The updated model weights are then broadcast to all remote workers.
- **used for policy gradient or on-policy training algos**
- step
    - sample from rollout workers `samples ~ [e.sample.remote() for e in self.workers.remote_workers()`
    - convert samples to MultiagentBatch `samples = MultiAgentBatch({DEFAULT_POLICY_ID: samples}, samples.count)`
    - loop over policies and get batch to train on 
    ```python
    for policy_id, policy in self.policies.items():
        batch = samples.policy_batches[policy_id]
        ...  # normalize each batch field
        for i in range(self.num_sgd_iter):
            ...
            for minibatch in self._minibatches(batch):
                ...
                self.workers.local_worker().learn_on_batch(
                                MultiAgentBatch({policy_id: minibatch}, minibatch.count)
                )
    ```
- _minibatches
    - takes in `samples` as SampleBatch
    - if not `self.sgd_minibatch_size` yield samples directly 
    - not implemented for MultiagentBatch
    - 


SyncReplayOptimizer
- reference: https://github.com/ray-project/ray/blob/d51583dbd6dc9c082764b9ec06349678aaa71078/rllib/optimizers/sync_replay_optimizer.py
- subclass `PolicyOptimizer`
- variant of the local sync optimizer that supports replay (for DQN)
- This optimizer requires that rollout workers return an additional "td_error" array in the info return of compute_gradients(). This error term will be used for sample prioritization
- **separate replay buffer for each agent**
```python
def new_buffer():
    return ReplayBuffer(buffer_size)
self.replay_buffers = collections.defaultdict(new_buffer)
```
- step 
    - sample batch by `batch = self.workers.local_worker().sample()`
    - convert to MultiagentBatch 
    - add to replay buffer 
    ```python
    for policy_id, s in batch.policy_batches.items():
        for row in s.rows():
            self.replay_buffers[policy_id].add(
                pack_if_needed(row["obs"]),
                row["actions"],
                row["rewards"],
                pack_if_needed(row["new_obs"]),
                row["dones"],
                weight=None
            )
    ```
    - train with `self._optimize()`
- _optimize
    - sample from replay buffer, call worker `learn_on_batch` using those samples
    ```python
    samples = self._replay()
    samples = self.before_learn_on_batch(
            samples,
            self.workers.local_worker().policy_map,
            self.train_batch_size)
    info_dict = self.workers.local_worker().learn_on_batch(samples)
    ```
- _replay
    - collect samples for each agent/policy
    ```python
    samples = {}
    for policy_id, replay_buffer in self.replay_buffers.items():
        # get sample indices 
        ...
        idxes = replay_buffer.sample_idxes(self.train_batch_size)
        # get samples (does unpacking if needed)
        (obses_t, actions, rewards, obses_tp1,
            dones) = replay_buffer.sample_with_idxes(idxes)
        weights = np.ones_like(rewards)
        batch_indexes = -np.ones_like(rewards)
        # construct sample batch 
        samples[policy_id] = SampleBatch({
            "obs": obses_t,
            "actions": actions,
            "rewards": rewards,
            "new_obs": obses_tp1,
            "dones": dones,
            "weights": weights,
            "batch_indexes": batch_indexes
        })
    # convert to multiagent batch 
    return MultiAgentBatch(samples, self.train_batch_size)
    ```
- **used for value based or off-policy training algos**


SyncBatchReplayOptimizer
- reference: https://github.com/ray-project/ray/blob/60d4d5e1aaa9fde3cf541ee335e284d05e75679c/rllib/optimizers/sync_batch_replay_optimizer.py
- subclass `PolicyOptimizer`
- **Variant of the sync replay optimizer that replays entire batches. This enables RNN support. Does not currently support prioritization**
- _optimize
    - get `samples` by iteratively call `random.choice(self.replay_buffer)` until `self.train_batch_size` is met
    - convert to SampleBatch 
    - train with `self.workers.local_worker().learn_on_batch(samples)`
- step 
    - get samples by `batches = [e.sample.remote() for e in self.workers.remote_workers()])`
    - for each batch in batches, convert to MultiagentBatch 
    - add each batch to replay buffer by `self.replay_buffer.append(batch)` (replay buffer is just a list)
    - train with `self._optimize`


<!---
##############################################################################
##############################################################################
##############################################################################
-->

## Sampling 

Samplers 
- reference: https://github.com/ray-project/ray/blob/d51583dbd6dc9c082764b9ec06349678aaa71078/rllib/evaluation/sampler.py 
- SyncSampler
    - key arguments: env, policies, policy_mapping_fn
    - key attributes: extra_batches (Queue), perf_stats, rollout_provider (_env_runner), metrics_queue (Queue)
    - get_data
        - keep calling `next` on `rollout_provider`, put rollout_metrics to `metrics_queue` and return data otherwise
    - get_metrics
        - keep calling `self.metrics_queue.get_nowait()._replace(perf_stats=self.perf_stats.get())` and append to list for return
    - get_extra_batches 
        - keep calling `self.extra_batches.get_nowait()` and append to list for return 
- AsyncSampler 
    - subclass to **threading.Thread**, overwrites method `run` (which calls `_run`)
    - _run
    ```python
    ...
    queue_putter = self.queue.put
    ...
    while not self.shutdown:
        # The timeout variable exists because apparently, if one worker
        # dies, the other workers won't die with it, unless the timeout is
        # set to some large number. This is an empirical observation.
        item = next(rollout_provider)
        if isinstance(item, RolloutMetrics):
            self.metrics_queue.put(item)
        else:
            queue_putter(item)
    ```
- _env_runner
    - implements the common experience collection logic
    - Args:
        - base_env (BaseEnv): env implementing BaseEnv.
        - extra_batch_callback (fn): function to send extra batch data to.
        - policies (dict): Map of policy ids to Policy instances.
        - policy_mapping_fn (func): Function that maps agent ids to policy ids. This is called when an agent first enters the environment. The agent is then "bound" to the returned policy for the episode.
        - unroll_length (int): Number of episode steps before `SampleBatch` is yielded. Set to infinity to yield complete episodes.
        - horizon (int): Horizon of the episode.
        - preprocessors (dict): Map of policy id to preprocessor for the observations prior to filtering.
        - obs_filters (dict): Map of policy id to filter used to process observations for the policy.
        - clip_rewards (bool): Whether to clip rewards before postprocessing.
        - pack (bool): Whether to pack multiple episodes into each batch. This guarantees batches will be exactly `unroll_length` in size.
        - clip_actions (bool): Whether to clip actions to the space range.
        - callbacks (dict): User callbacks to run on episode events.
        - tf_sess (Session|None): Optional tensorflow session to use for batching TF policy evaluations.
        - perf_stats (PerfStats): Record perf stats into this object.
        - soft_horizon (bool): Calculate rewards but don't reset the environment when the horizon is hit.
        - no_done_at_end (bool): Ignore the done=True at the end of the episode and instead record done=False.
    - Yields:
        - rollout (SampleBatch): Object containing state, action, reward, terminal condition, and other fields as dictated by `policy`.


SampleBatch
- reference: https://github.com/ray-project/ray/blob/d51583dbd6dc9c082764b9ec06349678aaa71078/rllib/policy/sample_batch.py
- concat_samples
    - concatenate samples (list of SampleBatch), with each data column (keys) concatenated
- rows, columns, return data rows or columns (each key specifies a data column)
- shuffle 
- slice
    - takes in `start`, `end` as integers
    - returns a slice of the row data of this batch.
- split_by_episode 
    - splits this batch's data by `eps_id`. returns list of SampleBatch, one per distinct episode.
    - by recurrently chunking/splitting based on `eps_id` key in data 


MultiAgentBatch
- batch of experiences from multiple policies in the environment
- policy_batches (dict): Mapping from policy id to a normal SampleBatch of experiences. Note that these batches may be of different length


SampleBatchBuilder
- reference: https://github.com/ray-project/ray/blob/d51583dbd6dc9c082764b9ec06349678aaa71078/rllib/evaluation/sample_batch_builder.py
- Wrapper around a dictionary with string keys and array-like values, e.g. {"obs": [1, 2, 3], "reward": [0, -1, 1]} is a batch of 3 samples
- `self.buffers = collections.defaultdict(list)`
- `self.unroll_id = 0`, disambiguates unrolls within a single episode
- add_values (dict as row), add_batch (dict of rows), 
- build_and_reset, returns a sample batch including all previously added values (make copy and clear buffer)


MultiAgentSampleBatchBuilder
- input data is per-agent, while output data is per-policy. there is an M:N mapping between agents and policies. We retain one local batch builder per agent. When an agent is done, then its local batch is appended into the corresponding policy batch for the agent's policy
- `policy_map (dict)`: Maps policy ids to policy instances
- `self.policy_builders = {k: SampleBatchBuilder() for k in policy_map.keys()}`
- postprocess_batch_so_far
    - takes in `episode`: current MultiAgentEpisode object or None
    - pushes the postprocessed per-agent batches onto the per-policy builders, clearing per-agent state


<!---
##############################################################################
##############################################################################
##############################################################################
-->

## Environment

VectorEnv
- reference: https://github.com/ray-project/ray/blob/d51583dbd6dc9c082764b9ec06349678aaa71078/rllib/env/vector_env.py
- supports batch evaluation
- attributes
    - action_space (gym.Space): Action space of individual envs.
    - observation_space (gym.Space): Observation space of individual envs.
    - num_envs (int): Number of envs in this vector env.
- methods: `vector_reset`, `vector_step`


RemoteVectorEnv
- reference: https://github.com/ray-project/ray/blob/d51583dbd6dc9c082764b9ec06349678aaa71078/rllib/env/remote_vector_env.py
- This provides dynamic batching of inference as observations are returned from the remote simulator actors. envs can be stepped synchronously or async
- `poll`, `send_actions`, `try_reset`, `stop`


BaseEnv
- reference: https://github.com/ray-project/ray/blob/d51583dbd6dc9c082764b9ec06349678aaa71078/rllib/env/base_env.py
- lowest-level env interface used by RLlib for sampling
- BaseEnv models multiple agents executing **asynchronously** in multiple environments. A call to `poll()` returns observations from ready agents **keyed by their environment and agent ids**, and actions for those agents can be sent back via `send_actions()`
- All other env types can be adapted to BaseEnv. RLlib handles these conversions internally in RolloutWorker, e.g.
    - gym.Env => rllib.VectorEnv => rllib.BaseEnv
    - rllib.MultiAgentEnv => rllib.BaseEnv
    - rllib.ExternalEnv => rllib.BaseEnv
- to_base_env
- poll 
    - The returns are two-level dicts mapping from env_id to a dict of agent_id to values. The number of agents and envs can vary over times
    - returns `obs (dict)`, `rewards (dict)`, `dones(dict)` (The special key "__all__" is used to indicate env termination), `infos (dict)`, `off_policy_actions (dict)`
    - for `off_policy_actions`, Agents may take off-policy actions. When that happens, there will be an entry in this dict that contains the taken action. There is no need to send_actions() for agents that have already chosen off-policy actions.
- send_actions
    - takes in `actions_dict`, actions values keyed by env_id and agent_id
    - Called to send actions back to running agents in this env. Actions should be sent for each ready agent that returned observations in the previous poll() call.
- stop
    - Releases all resources used, call `env.close` for each env (unwrapped)


MultiagentEnv
- reference: https://github.com/ray-project/ray/blob/d51583dbd6dc9c082764b9ec06349678aaa71078/rllib/env/multi_agent_env.py
- step
    - in `dones (dict)` returned, **__all__** will be true when all agents done is true 


ExternalEnv
- reference: https://github.com/ray-project/ray/blob/d51583dbd6dc9c082764b9ec06349678aaa71078/rllib/env/external_env.py
- subclass **threading.Thread**
- env that interfaces with external agents
- Unlike simulator envs, control is inverted. The environment queries the policy to obtain actions and logs observations and rewards for training. This is in contrast to gym.Env, where the algorithm drives the simulation through env.step() calls.
- supports both on-policy actions (through `self.get_action()`), and off-policy actions (through `self.log_action()`).
- env is thread-safe, but individual episodes must be executed serially
- run 
    - Your run loop should continuously:
        1. Call self.start_episode(episode_id)
        2. Call self.get_action(episode_id, obs) or self.log_action(episode_id, obs, action)
        3. Call self.log_returns(episode_id, reward)
        4. Call self.end_episode(episode_id, obs)
        5. Wait if nothing to do.
- start_episode
- get_action
    - takes in `episode_id`, `observation`
    - record an observation and get the on-policy action (by `episode.wait_for_action(observation)`).
- log_action
    - takes in `episode_id`, `observation`, `action`
    - calls `episode.log_action(observation, action)`
- log_returns 
    - record returns from the environment
    - do `episode.cur_reward += reward`
- end_episode
    - takes in `episode_id`, `observation`
    - do `self._finished.add(episode.episode_id)` and `episode.done(observation)`

_ExternalEnvEpisode
- Tracked state for each active episode
- `data_queue`, `action_queue`


<!---
##############################################################################
##############################################################################
##############################################################################
-->

## Override mechanisms 
reference: https://github.com/ray-project/ray/blob/d51583dbd6dc9c082764b9ec06349678aaa71078/rllib/utils/annotations.py

override 
- takes in cls name, check if method to override exists in cls, raise error if not 
PublicAPI
- jsut for annotation, stable interface, return input object itself 
DeveloperAPI
- jsut for annotation, nustable interface, return input object itself 


<!---
##############################################################################
##############################################################################
##############################################################################
-->

## Runner 
reference: https://ray.readthedocs.io/en/latest/_modules/ray/tune/tune.html#run_experiments

`run_experiments`( list of experiments )
-> call `run` for each (returns a trial)
-> `run` instantiates `TrialRunner`, loop over `runner.step()`
-> `step` calls `self._process_events()`, 
-> `_process_events` calls `self._process_trial(trial)`
-> `_process_trial`, takes in trial, calls, trial_executor is a `RayTrialExecutor`
-> `RayTrialExecutor`, in `_setup_remote_runner` inits logger by `trial.init_logger()`
reference: https://github.com/ray-project/ray/blob/e2ba8c1898f3e309b5e25d65bc81d80c26c78e62/python/ray/tune/ray_trial_executor.py
```python
result = self.trial_executor.fetch_result(trial)
...
trial.update_last_result(
    result, terminate=(decision == TrialScheduler.STOP))
```
-> `update_last_result`
- reference: https://github.com/ray-project/ray/blob/e2ba8c1898f3e309b5e25d65bc81d80c26c78e62/python/ray/tune/trial.py
```python 
result.update(trial_id=self.trial_id, done=terminate)
if self.experiment_tag:
    result.update(experiment_tag=self.experiment_tag)
if self.verbose and (terminate or time.time() - self.last_debug >
                        DEBUG_PRINT_INTERVAL):
    print("Result for {}:".format(self))
    print("  {}".format(pretty_print(result).replace("\n", "\n  ")))
    self.last_debug = time.time()
self.set_location(Location(result.get("node_ip"), result.get("pid")))
self.last_result = result
self.last_update_time = time.time()
self.result_logger.on_result(self.last_result)
for metric, value in flatten_dict(result).items():
    if isinstance(value, Number):
        if metric not in self.metric_analysis:
            self.metric_analysis[metric] = {
                "max": value,
                "min": value,
                "last": value
            }
        else:
            self.metric_analysis[metric]["max"] = max(
                value, self.metric_analysis[metric]["max"])
            self.metric_analysis[metric]["min"] = min(
                value, self.metric_analysis[metric]["min"])
            self.metric_analysis[metric]["last"] = value
```
-> `result_logger` instantiated in `init_logger`, 
```python
if not self.result_logger:
    if not self.logdir:
        self.logdir = Trial.create_logdir(str(self), self.local_dir)
    elif not os.path.exists(self.logdir):
        os.makedirs(self.logdir)

    self.result_logger = UnifiedLogger(
        self.config,
        self.logdir,
        trial=self,
        loggers=self.loggers,
        sync_function=self.sync_to_driver_fn)
```
-> `UnifiedLogger`, combines all loggers, `on_result` iteratively calls all `on_result` for each contained logger 
reference: https://github.com/ray-project/ray/blob/2965dc1b724010efcfb9a2709c01c650293f778a/python/ray/tune/logger.py 


<!---
##############################################################################
##############################################################################
##############################################################################
-->

## Question on sample sizes 
- **sample_batch_size** & **train_batch_size**, refer to number of transitions (tuples of obs, action, rew)
reference: https://github.com/ray-project/ray/blob/5d7afe8092f5521a8faa8cdfc916dcd9a5848023/rllib/optimizers/rollout.py
```python
def collect_samples(agents, sample_batch_size, num_envs_per_worker,
                    train_batch_size):
    """Collects at least train_batch_size samples, never discarding any."""

    num_timesteps_so_far = 0
    trajectories = []
    agent_dict = {}

    for agent in agents:
        fut_sample = agent.sample.remote()
        agent_dict[fut_sample] = agent

    while agent_dict:
        [fut_sample], _ = ray.wait(list(agent_dict))
        agent = agent_dict.pop(fut_sample)
        next_sample = ray_get_and_free(fut_sample)
        assert next_sample.count >= sample_batch_size * num_envs_per_worker
        num_timesteps_so_far += next_sample.count
        trajectories.append(next_sample)

        # Only launch more tasks if we don't already have enough pending
        pending = len(agent_dict) * sample_batch_size * num_envs_per_worker
        if num_timesteps_so_far + pending < train_batch_size:
            fut_sample2 = agent.sample.remote()
            agent_dict[fut_sample2] = agent

    return SampleBatch.concat_samples(trajectories)
```
- effective size of transitions (stored to buffer and used for training) is 
**sample_batch_size** * **num_envs_per_worker** & **train_batch_size** * **num_envs_per_worker**
- 1 iteration = sample_batch_size * num_workers * num_envs_per_worker


<!---
##############################################################################
##############################################################################
##############################################################################
-->

## Tensorboard filter runs with regex
- or with strings, |
- begin with, ^


<!---
##############################################################################
##############################################################################
##############################################################################
-->

## Render openai gym env over server
reference: https://gist.github.com/joschu/e42a050b1eb5cfbb1fdc667c3450467a
reference: https://hub.packtpub.com/openai-gym-environments-wrappers-and-monitors-tutorial/
- The code should be run in an X11 session with the OpenGL extension (GLX)
- The code should be started in an Xvfb virtual display
- You can use X11 forwarding in ssh connection
- xvfb-run -s "-screen 0 640x480x24" python 04_cartpole_random_monitor.py


reference: https://github.com/openai/gym/issues/462
- xvfb-run -s "-screen 0 1400x900x24" python <your_script.py> will create a fake X server for it. apt-get xvfb if you don't have it.


reference: https://github.com/2017-fall-DL-training-program/Setup_tutorial/blob/master/OpenAI-gym-install.md
- To save videos, we aslo need to install ffmpeg
sudo apt install ffmpeg

reference: https://gist.github.com/8enmann/931ec2a9dc45fde871d2139a7d1f2d78
- xvfb-run -s "-screen 0 1400x900x24"
- use ssh with -X to enable X11 forwarding or -Y

reference: https://medium.com/@DJVJallday/a-how-to-on-deep-reinforcement-learning-setup-aws-with-keras-tensorflow-openai-gym-and-jupyter-88bc0cc67e02
- Inside the screen, start a fake X server 
screen -S "openai"
xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" bash
- import tempfile
from gym import wrappers
tdir = tempfile.mkdtemp()
env = gym.make('FrozenLake-v0')
env = wrappers.Monitor(env, tdir, force=True)


<!---
##############################################################################
##############################################################################
##############################################################################
-->

## Ray Dashboard 
install from source and use **Node.js** 
reference: https://ray.readthedocs.io/en/latest/installation.html#optional-dashboard-support


<!---
##############################################################################
##############################################################################
##############################################################################
-->

## Ray actor 
actor is a stateful worker, methods of the actor are scheduled on that specific worker and can access and mutate the state of that worker.

convert python class to ray actor by **@ray.remote** decorator 

Actor processes will be terminated automatically when the initial actor handle goes out of scope in Python

Actor handles can be passed into other tasks.


<!---
##############################################################################
##############################################################################
##############################################################################
-->

## Policy 

Policy
- reference: https://github.com/ray-project/ray/blob/master/rllib/policy/policy.py
- takes in `observation_space`, `action_space`, `config`
- compute_actions
    - Arguments:
        - obs_batch (np.ndarray): batch of observations
        - state_batches (list): list of RNN state input batches, if any
        - prev_action_batch (np.ndarray): batch of previous action values
        - prev_reward_batch (np.ndarray): batch of previous rewards
        - info_batch (info): batch of info objects
        - episodes (list): MultiAgentEpisode for each obs in obs_batch.
            This provides access to all of the internal episode state,
            which may be useful for model-based or multiagent algorithms.
        kwargs: forward compatibility placeholder
    - Returns:
        - actions (np.ndarray): batch of output actions, with shape like
            [BATCH_SIZE, ACTION_SHAPE].
        - state_outs (list): list of RNN state output batches, if any, with
            shape like [STATE_SIZE, BATCH_SIZE].
        - info (dict): dictionary of extra feature batches, if any, with
            shape like {"f1": [BATCH_SIZE, ...], "f2": [BATCH_SIZE, ...]}.
- learn_on_batch 
    - takes in `samples`, call `compute_gradients` and `apply_gradients`


TorchPolicy 
- reference: https://github.com/ray-project/ray/blob/master/rllib/policy/torch_policy.py
- takes in `observation_space`, `action_space`, `model`, `loss`, `action_distribution_class`
    - model (nn.Module): PyTorch policy module. Given observations as
        input, this module must return a list of outputs where the
        first item is action logits, and the rest can be any value.
    - loss (func): Function that takes (policy, model, dist_class,
        train_batch) and returns a single scalar loss.
    - action_distribution_class (ActionDistribution): Class for action
        distribution. 
- optimizer, method that returns configurable 
- get_initial_state, calls `self.model.get_initial_state()` and turn to numpy arrays
- compute_actions
```python
 with torch.no_grad():
    input_dict = self._lazy_tensor_dict({
        "obs": obs_batch,
    })
    if prev_action_batch:
        input_dict["prev_actions"] = prev_action_batch
    if prev_reward_batch:
        input_dict["prev_rewards"] = prev_reward_batch
    model_out = self.model(input_dict, state_batches, [1])
    logits, state = model_out
    action_dist = self.dist_class(logits, self.model)
    actions = action_dist.sample()
    return (actions.cpu().numpy(), [h.cpu().numpy() for h in state],
            self.extra_action_out(input_dict, state_batches,
                                    self.model))
```
- learn_on_batch 
```python
train_batch = self._lazy_tensor_dict(postprocessed_batch)

loss_out = self._loss(self, self.model, self.dist_class, train_batch)
self._optimizer.zero_grad()
loss_out.backward()

grad_process_info = self.extra_grad_process()
self._optimizer.step()

grad_info = self.extra_grad_info(train_batch)
grad_info.update(grad_process_info)
return {LEARNER_STATS_KEY: grad_info}
```

build_torch_policy
- reference: https://github.com/ray-project/ray/blob/master/rllib/policy/torch_policy_template.py
- similar to `build_trainer`
- arguments:
```python
    name,
    loss_fn,
    get_default_config=None,
    stats_fn=None,
    postprocess_fn=None,
    extra_action_out_fn=None,
    extra_grad_process_fn=None,
    optimizer_fn=None,
    before_init=None,
    after_init=None,
    make_model_and_action_dist=None,
    mixins=None
```

<!---
##############################################################################
##############################################################################
##############################################################################
-->

## Recurrent 

off-policy DDPG 
-> off-policy PG could use important sampling to reweight gradients
-> DPG use deterministic policy, and because the deterministic policy gradient removes the integral over actions, we can avoid importance sampling
                      
key is in https://github.com/ray-project/ray/blob/d51583dbd6dc9c082764b9ec06349678aaa71078/rllib/evaluation/sampler.py 
SyncSampler will implicitly record hidden states 

rnn states stored in transition tuples as 
- state_in_{i}, state_out_{i}
where i is the index for hidden state in hidden state list (e.g. 2 for lstm)

chop_into_sequences 
reference: https://github.com/ray-project/ray/blob/60d4d5e1aaa9fde3cf541ee335e284d05e75679c/rllib/policy/rnn_sequencing.py





### on deterministic actions 
reference: https://github.com/shariqiqbal2810/maddpg-pytorch/blob/40388d7c18e4662cf23c826d97e209df9003d86c/algorithms/maddpg.py 
Forward pass as if onehot (hard=True) but backprop through a differentiable Gumbel-Softmax sample. The MADDPG paper uses the Gumbel-Softmax trick to backprop through discrete categorical samples, but I'm not sure if that is correct since it removes the assumption of a deterministic policy for DDPG. Regardless, discrete policies don't seem to learn properly without it.



<!---
##############################################################################
##############################################################################
##############################################################################
-->

## Acceleration

removing lz4 makes running much faster