# Custom implementation of Random Network Distillation
# https://blog.openai.com/reinforcement-learning-with-prediction-based-rewards/
#


import logging
import tensorflow as tf

from ray.rllib.agents import with_common_config
from ray.rllib.agents import ppo
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph
from ray.rllib.optimizers import SyncSamplesOptimizer, LocalMultiGPUOptimizer
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.evaluation.tf_policy_graph import TFPolicyGraph, LearningRateSchedule
from ray.rllib.models.misc import linear, normc_initializer
from ray.rllib.utils.explained_variance import explained_variance
from ray.rllib.evaluation.postprocessing import compute_advantages

logger = logging.getLogger(__name__)

# yapf: disable
# __sphinx_doc_begin__
DEFAULT_CONFIG = with_common_config({
    # If true, use the Generalized Advantage Estimator (GAE)
    # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
    "use_gae": True,
    # GAE(lambda) parameter
    "lambda": 1.0,
    # Initial coefficient for KL divergence
    "kl_coeff": 0.2,
    # Size of batches collected from each worker
    "sample_batch_size": 200,
    # Number of timesteps collected for each SGD round
    "train_batch_size": 4000,
    # Total SGD batch size across all devices for SGD (multi-gpu only)
    "sgd_minibatch_size": 128,
    # Number of SGD iterations in each outer loop
    "num_sgd_iter": 30,
    # Stepsize of SGD
    "lr": 5e-5,
    # Learning rate schedule
    "lr_schedule": None,
    # Share layers for value function
    "vf_share_layers": False,
    # Coefficient of the value function loss
    "vf_loss_coeff": 1.0,
    # Coefficient of the entropy regularizer
    "entropy_coeff": 0.0,
    # PPO clip parameter
    "clip_param": 0.3,
    # Clip param for the value function. Note that this is sensitive to the
    # scale of the rewards. If your expected V is large, increase this.
    "vf_clip_param": 10.0,
    # Target value for KL divergence
    "kl_target": 0.01,
    # Number of GPUs to use for SGD
    "num_gpus": 0,
    # Whether to allocate GPUs for workers (if > 0).
    "num_gpus_per_worker": 0,
    # Whether to allocate CPUs for workers (if > 0).
    "num_cpus_per_worker": 1,
    # Whether to rollout "complete_episodes" or "truncate_episodes"
    "batch_mode": "truncate_episodes",
    # Which observation filter to apply to the observation
    "observation_filter": "MeanStdFilter",
    # Use the sync samples optimizer instead of the multi-gpu one
    "simple_optimizer": False,
    # Size of the embedding space for Random Network Distillation predictions
    "embedding_size": 512
})
# __sphinx_doc_end__
# yapf: enable


class PPORNDLoss:
    def __init__(self,
                 action_space,
                 value_targets,
                 advantages_ext,
                 advantages_int,
                 actions,
                 logits,
                 vf_preds,
                 curr_action_dist,
                 value_fn,
                 cur_kl_coeff,
                 rnd_target,
                 rnd_predictor,
                 entropy_coeff=0,
                 clip_param=0.1,
                 vf_clip_param=0.1,
                 vf_loss_coeff=1.0,
                 use_gae=True,
                 rnd_pred_update_prop=0.25):
        """Constructs the loss for Proximal Policy Objective with Random Networks Distillation
        Arguments:
            action_space: Environment observation space specification.
            value_targets (Placeholder): Placeholder for target values; used
                for GAE.
            actions (Placeholder): Placeholder for actions taken
                from previous model evaluation.
            advantages_ext (Placeholder): Placeholder for calculated extrinsic advantages
                from previous model evaluation.
            advantages_int (Placeholder): Placeholder for calculated intrinsic advantages
                from previous model evaluation.
            logits (Placeholder): Placeholder for logits output from
                previous model evaluation.
            vf_preds (Placeholder): Placeholder for value function output
                from previous model evaluation.
            curr_action_dist (ActionDistribution): ActionDistribution
                of the current model.
            value_fn (Tensor): Current value function output Tensor.
            cur_kl_coeff (Variable): Variable holding the current PPO KL
                coefficient.
            rnd_target (Tensor): Current RND target network output Tensor
            rnd_predictor (Tensor): Current RND predictor network output Tensor
            entropy_coeff (float): Coefficient of the entropy regularizer.
            clip_param (float): Clip parameter
            vf_clip_param (float): Clip parameter for the value function
            vf_loss_coeff (float): Coefficient of the value function loss
            use_gae (bool): If true, use the Generalized Advantage Estimator.
            rnd_pred_update_prop (float): Proportion of experience used for RND predictor update.
        """
        dist_cls, _ = ModelCatalog.get_action_dist(action_space)
        prev_dist = dist_cls(logits)
        # Make loss functions.
        logp_ratio = tf.exp(
            curr_action_dist.logp(actions) - prev_dist.logp(actions))
        action_kl = prev_dist.kl(curr_action_dist)
        self.mean_kl = tf.reduce_mean(action_kl)

        curr_entropy = curr_action_dist.entropy()
        self.mean_entropy = tf.reduce_mean(curr_entropy)

        surrogate_loss = tf.minimum(
            advantages_ext * logp_ratio,
            advantages_ext * tf.clip_by_value(logp_ratio, 1 - clip_param,
                                          1 + clip_param))
        self.mean_policy_loss = tf.reduce_mean(-surrogate_loss)

        if use_gae:
            vf_loss1 = tf.square(value_fn - value_targets)
            vf_clipped = vf_preds + tf.clip_by_value(
                value_fn - vf_preds, -vf_clip_param, vf_clip_param)
            vf_loss2 = tf.square(vf_clipped - value_targets)
            vf_loss = tf.maximum(vf_loss1, vf_loss2)
            self.mean_vf_loss = tf.reduce_mean(vf_loss)
            loss = tf.reduce_mean(-surrogate_loss + cur_kl_coeff * action_kl +
                                  vf_loss_coeff * vf_loss -
                                  entropy_coeff * curr_entropy)
        else:
            self.mean_vf_loss = tf.constant(0.0)
            loss = tf.reduce_mean(-surrogate_loss + cur_kl_coeff * action_kl -
                                  entropy_coeff * curr_entropy)
        # TODO: add value loss for intrinsic rewards

        # Add RND loss terms to vf_loss
        # feat_var = tf.reduce_mean(tf.nn.moments(rnd_target, axes=[0])[1])  # TODO: use where?
        # max_feat = tf.reduce_max(tf.abs(rnd_target))  # TODO: use where?
        targets = tf.stop_gradient(rnd_target)
        self.int_rew = tf.reduce_mean(tf.square(targets - rnd_predictor), axis=-1, keep_dims=True)
        self.aux_loss = tf.reduce_mean(tf.square(targets - rnd_predictor), -1)
        mask = tf.random_uniform(shape=tf.shape(self.aux_loss), minval=0., maxval=1., dtype=tf.float32)
        mask = tf.cast(mask < rnd_pred_update_prop, tf.float32)
        self.aux_loss = tf.reduce_sum(mask * self.aux_loss) / tf.maximum(tf.reduce_sum(mask), 1.)
        loss = loss + self.aux_loss

        self.loss = loss


class PPORNDPolicyGraph(PPOPolicyGraph):
    def __init__(self,
                 observation_space,
                 action_space,
                 config,
                 existing_inputs=None):
        """
        Arguments:
            observation_space: Environment observation space specification.
            action_space: Environment action space specification.
            config (dict): Configuration values for PPORND graph.
            existing_inputs (list): Optional list of tuples that specify the
                placeholders upon which the graph should be built upon.
        """
        config = dict(DEFAULT_CONFIG, **config)
        self.sess = tf.get_default_session()
        self.action_space = action_space
        self.config = config
        self.kl_coeff_val = self.config["kl_coeff"]
        self.kl_target = self.config["kl_target"]
        dist_cls, logit_dim = ModelCatalog.get_action_dist(action_space)

        if existing_inputs:
            obs_ph, value_targets_ph, adv_ph, act_ph, \
                logits_ph, vf_preds_ph = existing_inputs[:6]
            # TODO: add adv_ph_int
            existing_state_in = existing_inputs[6:-1]
            existing_seq_lens = existing_inputs[-1]
        else:
            obs_ph = tf.placeholder(
                tf.float32,
                name="obs",
                shape=(None, ) + observation_space.shape)
            adv_ph = tf.placeholder(
                tf.float32, name="advantages", shape=(None, ))
            adv_int_ph = tf.placeholder(
                tf.float32, name="advantages_int", shape=(None, ))
            act_ph = ModelCatalog.get_action_placeholder(action_space)
            logits_ph = tf.placeholder(
                tf.float32, name="logits", shape=(None, logit_dim))
            vf_preds_ph = tf.placeholder(
                tf.float32, name="vf_preds", shape=(None, ))
            value_targets_ph = tf.placeholder(
                tf.float32, name="value_targets", shape=(None, ))
            existing_state_in = None
            existing_seq_lens = None
        self.observations = obs_ph

        self.loss_in = [
            ("obs", obs_ph),
            ("value_targets", value_targets_ph),
            ("advantages", adv_ph),
            ("actions", act_ph),
            ("logits", logits_ph),
            ("vf_preds", vf_preds_ph),
        ]
        self.model = ModelCatalog.get_model(
            obs_ph,
            logit_dim,
            self.config["model"],
            state_in=existing_state_in,
            seq_lens=existing_seq_lens)

        # KL Coefficient
        self.kl_coeff = tf.get_variable(
            initializer=tf.constant_initializer(self.kl_coeff_val),
            name="kl_coeff",
            shape=(),
            trainable=False,
            dtype=tf.float32)

        self.logits = self.model.outputs
        curr_action_dist = dist_cls(self.logits)
        self.sampler = curr_action_dist.sample()
        if self.config["use_gae"]:
            if self.config["vf_share_layers"]:
                self.value_function = tf.reshape(
                    linear(self.model.last_layer, 1, "value",
                           normc_initializer(1.0)), [-1])
            else:
                vf_config = self.config["model"].copy()
                # Do not split the last layer of the value function into
                # mean parameters and standard deviation parameters and
                # do not make the standard deviations free variables.
                vf_config["free_log_std"] = False
                vf_config["use_lstm"] = False
                with tf.variable_scope("value_function"):
                    self.value_function = ModelCatalog.get_model(
                        obs_ph, 1, vf_config).outputs
                    self.value_function = tf.reshape(self.value_function, [-1])
        else:
            self.value_function = tf.zeros(shape=tf.shape(obs_ph)[:1])

        # TODO: add another head in the policy network for estimating value of intrinsic reward

        # RND target network
        with tf.variable_scope("rnd_target"):
            modelconfig = self.config["model"].copy()
            modelconfig["free_log_std"] = False
            modelconfig["use_lstm"] = False
            self.rnd_target = ModelCatalog.get_model(obs_ph, self.config["embedding_size"], modelconfig).outputs
            # self.rnd_target = tf.reshape(self.rnd_target, [-1])  # TODO: necessary?

        # RND predictor network
        with tf.variable_scope("rnd_predictor"):
            modelconfig = self.config["model"].copy()
            modelconfig["free_log_std"] = False
            modelconfig["use_lstm"] = False
            self.rnd_predictor = ModelCatalog.get_model(obs_ph, self.config["embedding_size"], modelconfig).outputs

        self.loss_obj = PPORNDLoss(
            action_space,
            value_targets_ph,
            adv_ph,
            adv_int_ph,
            act_ph,
            logits_ph,
            vf_preds_ph,
            curr_action_dist,
            self.value_function,
            self.kl_coeff,
            self.rnd_target,
            self.rnd_predictor,
            # TODO: valid_mask??
            entropy_coeff=self.config["entropy_coeff"],
            clip_param=self.config["clip_param"],
            vf_clip_param=self.config["vf_clip_param"],
            vf_loss_coeff=self.config["vf_loss_coeff"],
            use_gae=self.config["use_gae"])

        entropy_coeff = 0,
        clip_param = 0.1,
        vf_clip_param = 0.1,
        vf_loss_coeff = 1.0,
        use_gae = True,
        rnd_pred_update_prop = 0.25

        LearningRateSchedule.__init__(self, self.config["lr"],
                                      self.config["lr_schedule"])
        TFPolicyGraph.__init__(
            self,
            observation_space,
            action_space,
            self.sess,
            obs_input=obs_ph,
            action_sampler=self.sampler,
            loss=self.loss_obj.loss,
            loss_inputs=self.loss_in,
            state_inputs=self.model.state_in,
            state_outputs=self.model.state_out,
            seq_lens=self.model.seq_lens,
            max_seq_len=config["model"]["max_seq_len"])

        self.sess.run(tf.global_variables_initializer())
        self.explained_variance = explained_variance(value_targets_ph,
                                                     self.value_function)
        self.stats_fetches = {
            "cur_lr": tf.cast(self.cur_lr, tf.float64),
            "total_loss": self.loss_obj.loss,
            "policy_loss": self.loss_obj.mean_policy_loss,
            "vf_loss": self.loss_obj.mean_vf_loss,
            "vf_explained_var": self.explained_variance,
            "kl": self.loss_obj.mean_kl,
            "entropy": self.loss_obj.mean_entropy
        }

    def postprocess_trajectory(self, sample_batch, other_agent_batches=None):
        """Postprocesses a trajectory to compute (extrinsic) advantages and intrinsic advantages"""
        completed = sample_batch["dones"][-1]
        if completed:
            last_r = 0.0
        else:
            next_state = []
            for i in range(len(self.model.state_in)):
                next_state.append([sample_batch["state_out_{}".format(i)][-1]])
            last_r = self.value(sample_batch["new_obs"][-1], *next_state)
        # Extrinsic advantages computation
        batch = compute_advantages(
            sample_batch,
            last_r,
            self.config["gamma"],
            self.config["lambda"],
            use_gae=self.config["use_gae"])
        # Intrinsic advantages computation
        # TODO: for this to work we need to fill in vf_preds with the values from the new head
        batch_int = compute_advantages(
            sample_batch,
            last_r,
            self.config["gamma_int"],
            self.config["lambda_int"],
            use_gae=self.config["use_gae"])
        for b, bi in zip(batch, batch_int):
            b["advantages_int"] = bi["advantages"]
        return batch


class PPORNDAgent(ppo.PPOAgent):
    """Extension of PPO to incorporate curiosity via Random Network Distillation"""
    _agent_name = "PPORND"
    _default_config = DEFAULT_CONFIG
    _policy_graph = PPORNDPolicyGraph

    def _init(self):
        self._validate_config()
        self.local_evaluator = self.make_local_evaluator(
            self.env_creator, self._policy_graph)
        self.remote_evaluators = self.make_remote_evaluators(
            self.env_creator, self._policy_graph, self.config["num_workers"], {
                "num_cpus": self.config["num_cpus_per_worker"],
                "num_gpus": self.config["num_gpus_per_worker"]
            })
        if self.config["simple_optimizer"]:
            self.optimizer = SyncSamplesOptimizer(
                self.local_evaluator, self.remote_evaluators, {
                    "num_sgd_iter": self.config["num_sgd_iter"],
                    "train_batch_size": self.config["train_batch_size"],
                })
        else:
            self.optimizer = LocalMultiGPUOptimizer(
                self.local_evaluator, self.remote_evaluators, {
                    "sgd_batch_size": self.config["sgd_minibatch_size"],
                    "num_sgd_iter": self.config["num_sgd_iter"],
                    "num_gpus": self.config["num_gpus"],
                    "train_batch_size": self.config["train_batch_size"],
                    "standardize_fields": ["advantages"],
            })

    def _validate_config(self):
        waste_ratio = (
            self.config["sample_batch_size"] * self.config["num_workers"] /
            self.config["train_batch_size"])
        if waste_ratio > 1:
            msg = ("sample_batch_size * num_workers >> train_batch_size. "
                   "This means that many steps will be discarded. Consider "
                   "reducing sample_batch_size, or increase train_batch_size.")
            if waste_ratio > 1.5:
                raise ValueError(msg)
            else:
                logger.warn(msg)
        if self.config["sgd_minibatch_size"] > self.config["train_batch_size"]:
            raise ValueError(
                "Minibatch size {} must be <= train batch size {}.".format(
                    self.config["sgd_minibatch_size"],
                    self.config["train_batch_size"]))
        if (self.config["batch_mode"] == "truncate_episodes"
            and not self.config["use_gae"]):
            raise ValueError(
                "Episode truncation is not supported without a value function")

    #  def _train(self):  # TODO maybe override this method to change how training works?
