from mindspore import nn, ops

def get_experts(n_experts):
    def expert():
        return nn.SequentialCell([
            nn.Dense(28*28, 64, activation='relu'),
            nn.Dense(64, 128, activation='relu'),
            nn.Dense(128, 256, activation='relu'),
            nn.Dense(256, 10, activation='softmax')
        ])
    return [expert() for _ in range(n_experts)]

def get_gate(n_expert):
    return nn.SequentialCell([
        nn.Dense(28*28, 64, activation='relu'),
        nn.Dense(64, 128, activation='relu'),
        nn.Dense(n_expert, activation='softmax')
    ])

class MoE(nn.Cell):
    def __init__(self, n_expert):
        super().__init__()
        self.experts = nn.CellList(get_experts(n_expert))
        self.gate = get_gate(n_expert)
    
    def construct(self, inputs):
        gates = self.gate(inputs)
        gates = gates.expand_dims(-1)
        values = ops.stack([expert(inputs) for expert in self.experts], axis=-1)
        return ops.matmul(values, gates)