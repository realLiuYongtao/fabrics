from graphviz import Digraph

# 绘制ChannelAttention模块
def draw_channel_attention():
    dot = Digraph(name='ChannelAttention', format='pdf', graph_attr={'rankdir': 'TB'})

    # 输入节点
    dot.node('x', 'Input\n(B, C, L)', shape='box')

    # 平均池化分支
    with dot.subgraph(name='avg_branch') as c:
        c.node('avg_pool', 'AdaptiveAvgPool1d(1)\n(B, C, 1)', shape='box')
        c.node('avg_squeeze', 'squeeze(-1)\n(B, C)', shape='box')
        c.node('avg_fc', 'MLP\n(B, C)', shape='box')
        c.node('avg_unsqueeze', 'unsqueeze(-1)\n(B, C, 1)', shape='box')
        c.edge('avg_pool', 'avg_squeeze')
        c.edge('avg_squeeze', 'avg_fc')
        c.edge('avg_fc', 'avg_unsqueeze')

    # 最大池化分支
    with dot.subgraph(name='max_branch') as c:
        c.node('max_pool', 'AdaptiveMaxPool1d(1)\n(B, C, 1)', shape='box')
        c.node('max_squeeze', 'squeeze(-1)\n(B, C)', shape='box')
        c.node('max_fc', 'MLP\n(B, C)', shape='box')
        c.node('max_unsqueeze', 'unsqueeze(-1)\n(B, C, 1)', shape='box')
        c.edge('max_pool', 'max_squeeze')
        c.edge('max_squeeze', 'max_fc')
        c.edge('max_fc', 'max_unsqueeze')

    # 合并与输出
    dot.node('add', 'add\n(B, C, 1)', shape='circle')
    dot.node('sigmoid', 'Sigmoid\n(B, C, 1)', shape='box')
    dot.node('out', 'Output\n(B, C, L)', shape='box')

    # 连接边
    dot.edge('x', 'avg_pool')
    dot.edge('x', 'max_pool')
    dot.edge('avg_unsqueeze', 'add')
    dot.edge('max_unsqueeze', 'add')
    dot.edge('add', 'sigmoid')
    dot.edge('sigmoid', 'out', label=' * (element-wise)')
    dot.edge('x', 'out', style='dashed')

    dot.render('channel_attention', view=True)

draw_channel_attention()