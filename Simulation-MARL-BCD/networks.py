import torch as T
import torch.nn as nn
import torch.nn.functional as F
import os


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, fc3_dims, n_agents,
                 action_dim, name, agent_label, chkpt_dir='model2/ris_sac_model'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.action_dim = action_dim
        self.name = name + '_' + str(agent_label)
        self.checkpoint_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), chkpt_dir)
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Determine the input dimension based on whether it's a global or local critic
        if 'global' in self.name:
            # Global critic sees all states and all actions
            input_size = self.input_dims * n_agents + self.action_dim
        else:
            # Local critic sees its own state and its own action
            input_size = self.input_dims + self.action_dim

        self.fc1 = nn.Linear(input_size, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.q = nn.Linear(self.fc3_dims, 1)

        self.optimizer = T.optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        # Concatenate state and action for the input layer
        x = T.cat([state, action], dim=1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        q_value = self.q(x)
        return q_value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))
