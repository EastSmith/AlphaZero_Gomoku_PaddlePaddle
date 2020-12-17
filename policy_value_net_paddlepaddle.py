import paddle.fluid as fluid
import numpy as np
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph import Conv2D, Linear



class Net(fluid.dygraph.Layer):
    def __init__(self,board_width, board_height):
        super(Net, self).__init__()
        self.board_width = board_width
        self.board_height = board_height
        # common layers
        self.conv1 = Conv2D(num_channels=4,num_filters=32,filter_size=3,padding=1,act="relu")
        self.conv2 = Conv2D(num_channels=32,num_filters=64,filter_size=3,padding=1,act="relu")
        self.conv3 = Conv2D(num_channels=64,num_filters=128,filter_size=3,padding=1,act="relu")
        # action policy layers
        self.act_conv1 = Conv2D(num_channels=128,num_filters=4,filter_size=1,padding=0,act="relu")
        self.act_fc1 = Linear(4*self.board_width*self.board_height,
                                 self.board_width*self.board_height)
        self.val_conv1 = Conv2D(num_channels=128,num_filters=2,filter_size=1,padding=0,act="relu")
        self.val_fc1 = Linear(2*self.board_width*self.board_height, 64,act="relu")
        self.val_fc2 = Linear(64, 1,act="tanh")

    def forward(self, inputs):
        # common layers 
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        # action policy layers
        x_act = self.act_conv1(x)
        x_act = fluid.layers.reshape(
                x_act, [-1, 4 * self.board_height * self.board_width])
        x_act = self.act_fc1(x_act)
        x_act  = fluid.layers.log(fluid.layers.softmax(x_act))        
        # state value layers
        x_val  = self.val_conv1(x)
        x_val = fluid.layers.reshape(
                x_val, [-1, 2 * self.board_height * self.board_width])
        x_val = self.val_fc1(x_val)
        x_val = self.val_fc2(x_val)

        return x_act,x_val

class PolicyValueNet():
    """policy-value network """
    def __init__(self, board_width, board_height,
                 model_file=None, use_gpu=True):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty
        # the policy value net module
        # place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()

        self.policy_value_net = Net(self.board_width, self.board_height)        
        
        self.optimizer  = fluid.optimizer.Adam(learning_rate=0.02,
                                parameter_list=self.policy_value_net.parameters(),
                                regularization= fluid.regularizer.L2Decay(regularization_coeff=0.001)
                                     )

        if model_file:
            net_params, _ = fluid.dygraph.load_dygraph(model_file)
            self.policy_value_net.load_dict(net_params)
            
    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        # state_batch = fluid.dygraph.to_variable(np.ndarray(state_batch))
        state_batch = fluid.dygraph.to_variable(state_batch)
        log_act_probs, value = self.policy_value_net(state_batch)
        act_probs = np.exp(log_act_probs.numpy())
        return act_probs, value.numpy()

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, 4, self.board_width, self.board_height)).astype("float32")
        
        # print(current_state.shape)
        current_state = fluid.dygraph.to_variable(current_state)
        log_act_probs, value = self.policy_value_net(current_state)
        act_probs = np.exp(log_act_probs.numpy().flatten())
        
        act_probs = zip(legal_positions, act_probs[legal_positions])
        # value = value.numpy()
        return act_probs, value.numpy()

    def train_step(self, state_batch, mcts_probs, winner_batch, lr=0.002):
        """perform a training step"""
        # wrap in Variable
        state_batch = fluid.dygraph.to_variable(state_batch)
        mcts_probs = fluid.dygraph.to_variable(mcts_probs)
        winner_batch = fluid.dygraph.to_variable(winner_batch)

        # zero the parameter gradients
        self.optimizer.clear_gradients()
        # set learning rate
        self.optimizer  = fluid.optimizer.Adam(lr,
                                parameter_list=self.policy_value_net.parameters(),
                                regularization= fluid.regularizer.L2Decay(regularization_coeff=0.001)
                                     )

        # forward
        log_act_probs, value = self.policy_value_net(state_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value = fluid.layers.reshape(x=value, shape=[-1])
        value_loss = fluid.layers.mse_loss(input=value, label=winner_batch)
        policy_loss = -fluid.layers.reduce_mean(fluid.layers.reduce_sum(mcts_probs*log_act_probs, dim=1))
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.optimizer.minimize(loss)
        # calc policy entropy, for monitoring only
        entropy = -fluid.layers.reduce_mean(
                fluid.layers.reduce_sum(fluid.layers.exp(log_act_probs) * log_act_probs, dim=1)
                )
        return loss.numpy(), entropy.numpy()[0]    

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        fluid.save_dygraph(net_params, model_file)

# if __name__ == '__main__':
#     with fluid.dygraph.guard():
#         img = np.random.rand(3, 4, 15, 15).astype("float32")
#         # img = fluid.dygraph.to_variable(img)
#         outs = PolicyValueNet(15,15).policy_value(img)[1]
#         print(outs.shape,outs)

#         x1 = np.random.rand(3, 4, 15, 15).astype("float32")
#         x2 = np.random.rand(3, 225).astype("float32")
#         x3 = np.random.rand(3).astype("float32")

        
#         e = PolicyValueNet(15,15).train_step(x1,x2,x3)[1]
#         print(e,e.shape)
