import paddle
import numpy as np

import paddle.nn as nn 
import paddle.nn.functional as F



class Net(paddle.nn.Layer):
    def __init__(self,board_width, board_height):
        super(Net, self).__init__()
        self.board_width = board_width
        self.board_height = board_height
        # 公共网络层
        self.conv1 = nn.Conv2D(in_channels=4,out_channels=32,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2D(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2D(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        # 行动策略网络层
        self.act_conv1 = nn.Conv2D(in_channels=128,out_channels=4,kernel_size=1,padding=0)
        self.act_fc1 = nn.Linear(4*self.board_width*self.board_height,
                                 self.board_width*self.board_height)
        self.val_conv1 = nn.Conv2D(in_channels=128,out_channels=2,kernel_size=1,padding=0)
        self.val_fc1 = nn.Linear(2*self.board_width*self.board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, inputs):
        # 公共网络层 
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # 行动策略网络层
        x_act = F.relu(self.act_conv1(x))
        x_act = paddle.reshape(
                x_act, [-1, 4 * self.board_height * self.board_width])
        
        x_act  = F.log_softmax(self.act_fc1(x_act))        
        # 状态价值网络层
        x_val  = F.relu(self.val_conv1(x))
        x_val = paddle.reshape(
                x_val, [-1, 2 * self.board_height * self.board_width])
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))

        return x_act,x_val

class PolicyValueNet():
    """策略&值网络 """
    def __init__(self, board_width, board_height,
                 model_file=None, use_gpu=True):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-3  # coef of l2 penalty
        

        self.policy_value_net = Net(self.board_width, self.board_height)        
        
        self.optimizer  = paddle.optimizer.Adam(learning_rate=0.02,
                                parameters=self.policy_value_net.parameters(), weight_decay=self.l2_const)
                                     

        if model_file:
            net_params = paddle.load(model_file)
            self.policy_value_net.set_state_dict(net_params)
            
    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        # state_batch = paddle.to_tensor(np.ndarray(state_batch))
        state_batch = paddle.to_tensor(state_batch)
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
        current_state = paddle.to_tensor(current_state)
        log_act_probs, value = self.policy_value_net(current_state)
        act_probs = np.exp(log_act_probs.numpy().flatten())
        
        act_probs = zip(legal_positions, act_probs[legal_positions])
        # value = value.numpy()
        return act_probs, value.numpy()

    def train_step(self, state_batch, mcts_probs, winner_batch, lr=0.002):
        """perform a training step"""
        # wrap in Variable
        state_batch = paddle.to_tensor(state_batch)
        mcts_probs = paddle.to_tensor(mcts_probs)
        winner_batch = paddle.to_tensor(winner_batch)

        # zero the parameter gradients
        self.optimizer.clear_gradients()
        # set learning rate
        self.optimizer.set_lr(lr)

                                     

        # forward
        log_act_probs, value = self.policy_value_net(state_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value = paddle.reshape(x=value, shape=[-1])
        value_loss = F.mse_loss(input=value, label=winner_batch)
        policy_loss = -paddle.mean(paddle.sum(mcts_probs*log_act_probs, axis=1))
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.optimizer.minimize(loss)
        # calc policy entropy, for monitoring only
        entropy = -paddle.mean(
                paddle.sum(paddle.exp(log_act_probs) * log_act_probs, axis=1)
                )
        return loss.numpy(), entropy.numpy()[0]    

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        paddle.save(net_params, model_file)
