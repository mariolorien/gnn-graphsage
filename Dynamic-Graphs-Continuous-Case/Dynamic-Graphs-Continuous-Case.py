import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import GRUCell, Linear, Sequential, ReLU

try:
    from torch_geometric.data import TemporalData
except ImportError:
    class TemporalData:
        """
        Small fallback container in case torch_geometric TemporalData
        is not available in the environment.
        """
        def __init__(self, src, dst, t, msg):
            self.src = src
            self.dst = dst
            self.t = t
            self.msg = msg

        def __repr__(self):
            return (
                f"TemporalData(src_shape={tuple(self.src.shape)}, "
                f"dst_shape={tuple(self.dst.shape)}, "
                f"t_shape={tuple(self.t.shape)}, "
                f"msg_shape={tuple(self.msg.shape)})"
            )


"""
Continuous-Time Dynamic Graph (CTDG) from CSV

This version does not generate artificial event data inside the script.
Instead, it loads a node table and an event table from CSV files.

Idea:
- nodes.csv gives us the nodes and their static features
We can interpret those however we like, for example:

feature_1 = profile score
feature_2 = months in system
feature_3 = manager flag

- events.csv gives us the temporal interaction log

src = source node
dst = destination node
timestamp = real datetime
the last 4 columns are the event/message features

So in this example:

msg_dim = 4
because each event has 4 numeric attributes:
event_weight
channel_code
priority
response_time

- we convert that into a TemporalData object
- then we run a simple memory-based CTDG model

This is closer to how a real project would look.
"""
"""
What is a TemporalData object? 

Is a PyTorch geometric object designed to store a stream of time-stamped interactions. 

For a normal static graph in PyG we use the Data object, which store things like 
- x for node features
- edge_index for graph connections 
- y for labels

For a continuous-time graph that static format is not enough, because the key thing is the 
event sequence over time. 

Example: 

Suppose that we have 3 events: 

A interacts with B at time 1.0
B interacts with C at time 2.5
A interacts with C at time 4.0

After converting node IDs to indices, you might have:

src = tensor([0, 1, 0])
dst = tensor([1, 2, 2])
t   = tensor([1.0, 2.5, 4.0])
msg = tensor([
    [0.7, 1.0],
    [1.2, 0.0],
    [0.4, 1.0]
])

Then, 

data = TemporalData(src=src, dst=dst, t=t, msg=msg)

means:

this object now stores the whole temporal interaction history

A TemporalData object is not the model.

It is just the data structure holding the temporal graph information.

So:

TemporalData = the dataset container
CTDGModel = the neural network that learns from it
"""

# ------------------------------------------------------------------
# PART 1: LOAD CTDG DATA FROM CSV
# ------------------------------------------------------------------

def load_continuous_time_event_graph_from_csv(
    nodes_csv_path,
    events_csv_path,
    node_id_col="node_id",
    src_col="src",
    dst_col="dst",
    time_col="timestamp",
    node_feature_cols=None,
    event_feature_cols=None,
    label_col=None,
):
    """
    Load a continuous-time event graph from CSV files.

    Parameters
    ----------
    nodes_csv_path : str
        Path to the node CSV

    events_csv_path : str
        Path to the event CSV

    node_id_col : str
        Name of the node ID column in nodes.csv

    src_col : str
        Name of the source-node column in events.csv

    dst_col : str
        Name of the destination-node column in events.csv

    time_col : str
        Name of the timestamp column in events.csv

    node_feature_cols : list[str]
        Names of numeric node feature columns in nodes.csv

    event_feature_cols : list[str]
        Names of numeric event feature columns in events.csv

    label_col : str or None
        Optional node label column in nodes.csv.
        If not provided, labels are derived from event participation count.

    Returns
    -------
    data : TemporalData
        TemporalData object with extra attributes:
        - x
        - y
        - event_counts
        - node_ids
        - node_id_to_index
    """

    if node_feature_cols is None:
        raise ValueError("Please provide node_feature_cols.")

    if event_feature_cols is None:
        raise ValueError("Please provide event_feature_cols.")

    # ------------------------------------------------------------
    # LOAD CSV FILES
    # ------------------------------------------------------------
    nodes_df = pd.read_csv(nodes_csv_path)
    events_df = pd.read_csv(events_csv_path)

    # ------------------------------------------------------------
    # BUILD NODE ID -> INDEX MAPPING
    # ------------------------------------------------------------
    """
    Temporal models work with integer node indices:
    0, 1, 2, ..., N-1

    But in real CSV files, node IDs may be strings such as:
    A, B, C, user_101, etc.

    So we create a mapping.
    
    For example, if we have node features in:

        x.shape = [num_nodes, feature_dim]

        then:

        x[0] = features of node 0
        x[1] = features of node 1
        x[2] = features of node 2

        PyTorch expects that kind of indexing. We cannot do:

        x["A"]

        in the same way.

        So what is the mapping?

        Suppose our CSV has node IDs like:

        node_id
        A
        B
        C
        D

        We convert them into integers such as:

        A -> 0
        B -> 1
        C -> 2
        D -> 3
        
        The mapping does not change who the node is.
        It only changes how the node is referred to internally.

        So:

        "A" is the real label from the dataset
        0 is the internal index used by the model
        Tiny analogy

        Think of a school register:

        student name = "Mario"
        internal seat number = 12

        The person is the same.
        The seat number just makes classroom organization easier.
        
    """
    node_ids = nodes_df[node_id_col].tolist()
    
    node_id_to_index = {}

    for idx, node_id in enumerate(node_ids):
        node_id_to_index[node_id] = idx

    """
    What the line above does? 
    
    If:

    node_ids = ["A", "B", "C", "D"]

    then enumerate(node_ids) gives:

    (0, "A")
    (1, "B")
    (2, "C")
    (3, "D")

    Then the loop builds the dictionary step by step:   
    
    Iteration 1
        idx = 0
        node_id = "A"

    Dictionary becomes:

        {"A": 0}
    
    Iteration 2
        idx = 1
        node_id = "B"

    Dictionary becomes:

        {"A": 0, "B": 1}
    
    """
    # ------------------------------------------------------------
    # KEEP ONLY EVENTS WHERE BOTH NODES EXIST IN nodes.csv
    # ------------------------------------------------------------
    events_df = events_df[
        events_df[src_col].isin(node_id_to_index)
        & events_df[dst_col].isin(node_id_to_index)
    ].copy()

    # ------------------------------------------------------------
    # CONVERT TIMESTAMP TO NUMERIC TIME
    # ------------------------------------------------------------
    """
    CTDG models need numeric time.

    If the CSV contains real datetimes, we convert them into
    seconds since the first event.
    """
    parsed_time = pd.to_datetime(events_df[time_col])
    events_df["_time_seconds"] = (parsed_time - parsed_time.min()).dt.total_seconds()

    # ------------------------------------------------------------
    # SORT EVENTS IN CHRONOLOGICAL ORDER
    # ------------------------------------------------------------
    events_df = events_df.sort_values("_time_seconds").reset_index(drop=True)

    # ------------------------------------------------------------
    # NODE FEATURES
    # ------------------------------------------------------------
    x = torch.tensor(
        nodes_df[node_feature_cols].to_numpy(dtype=float),
        dtype=torch.float
    )
    """
    What just happened here? 
    
    1. nodes_df[node_feature_cols]

    This selects only the columns listed in node_feature_cols.

    So if:

    node_feature_cols = ["feature_1", "feature_2", "feature_3"]

    and nodes_df looks like:

    node_id	feature_1	feature_2	feature_3
    A	   0.82	          36	         1
    B	   0.45	          12	         0
    C	   0.67	          24	         0

    then:

    nodes_df[node_feature_cols]

    gives just:

    feature_1	feature_2	feature_3
    0.82	     36	           1
    0.45	     12	           0
    0.67	     24	           0

    So it removes the node_id column and keeps only the numeric features.

    2. .to_numpy(dtype=float)

    This converts that pandas table into a NumPy array of floats.

    So it becomes something like:

    array([
        [0.82, 36.0, 1.0],
        [0.45, 12.0, 0.0],
        [0.67, 24.0, 0.0]
    ])

    This is now plain numeric data, no longer a DataFrame.

    3. torch.tensor(..., dtype=torch.float)

    This converts the NumPy array into a PyTorch tensor.

    So the final result is:

    tensor([
        [ 0.82, 36.00,  1.00],
        [ 0.45, 12.00,  0.00],
        [ 0.67, 24.00,  0.00]
    ])

    and this gets stored in x.
    
    
    """
    # ------------------------------------------------------------
    # EVENT STREAM
    # Here, from the csv file we extract the features that we will need 
    # to create our TemporalData object: src, dst, t, and msg. 
    # ------------------------------------------------------------
    src = torch.tensor(
        [node_id_to_index[node] for node in events_df[src_col]],
        dtype=torch.long
    )

    dst = torch.tensor(
        [node_id_to_index[node] for node in events_df[dst_col]],
        dtype=torch.long
    )

    t = torch.tensor(
        events_df["_time_seconds"].to_numpy(dtype=float),
        dtype=torch.float
    )

    msg = torch.tensor(
        events_df[event_feature_cols].to_numpy(dtype=float),
        dtype=torch.float
    )

    # ------------------------------------------------------------
    # NODE LABELS
    # ------------------------------------------------------------
    """
    Option 1:
    If a label column exists in nodes.csv, use it.

    Option 2:
    Otherwise derive a simple toy label:
    y = 1 if node participates in at least the average number of events
    y = 0 otherwise
    
    We do this because supervised models need labels; 
    in real life we would prefer real labels if they exist, but in a teaching or demo setting 
    we often create a simple artificial label so the temporal GNN has something to learn.
    """
    event_counts = torch.zeros(len(node_ids), dtype=torch.long)

    for s, d in zip(src, dst):
        event_counts[s] += 1
        event_counts[d] += 1

    if label_col is not None and label_col in nodes_df.columns:
        y = torch.tensor(nodes_df[label_col].to_numpy(), dtype=torch.long)
    else:
        avg_count = event_counts.float().mean()
        y = (event_counts.float() >= avg_count).long()

    # ------------------------------------------------------------
    # BUILD TEMPORAL DATA OBJECT
    # ------------------------------------------------------------
    data = TemporalData(src=src, dst=dst, t=t, msg=msg)
    data.x = x
    data.y = y
    data.event_counts = event_counts
    data.node_ids = node_ids
    data.node_id_to_index = node_id_to_index

    return data


# ------------------------------------------------------------------
# PART 2: CTDG MODEL
# ------------------------------------------------------------------

class CTDGModel(torch.nn.Module):
    """
    Simple memory-based Continuous-Time Dynamic Graph model.

    For each node:
    - start with an initial memory vector

    For each event:
    - read source memory
    - read destination memory
    - read event message
    - read time gaps
    - build an event embedding
    - update node memories

    Final step:
    - classify each node using its final memory
    """

    def __init__(self, in_channels, msg_dim, memory_dim, num_classes):
        super().__init__()

        """
        Convert static node features into initial memory states.
        in_channels means how many input features each node has at the start 
        if data.x.shape = [num_nodes, 3] than in_channels = 3
        
        Is asking: How wide is the node features before temporal learning starts? 
        
        memory_dim is then the learned temporal memory size. 
        In our CTDG model, each node has a memory that evolves over times as events happne. 
        If say memory_dim=8, then every node carries and 8 dimensional hidden state. 
        
        We do not call it embedding (like in the static cases) because this is not a 
        static representation. It updates after events.  So, it behaves like a memory/state: 
        
        - before event 1, node A has one hidden state
        - after event 1, node A’s hidden state changes
        - after event 2, it changes again
        - and so on
        
        When we say:

        memory_dim = 8

        we mean that each node has a vector like:

        [h1,h2,h3,h4,h5,h6,h7,h8]

        So each node is represented internally by 8 numbers.
        
        
        msg_dim asks how many features each event message has:
        
        In our case 4. 
        
        """
        self.node_encoder = Linear(in_channels, memory_dim)

        """
        Event encoder:
        input =
            source memory
            destination memory
            event message
            delta time for source
            delta time for destination
        output =
            one event embedding
        """
        self.event_mlp = Sequential(
            Linear(2 * memory_dim + msg_dim + 2, memory_dim),
            ReLU(),
            Linear(memory_dim, memory_dim)
        )

        """
        In the line above, sequential means apply the following layers one after the other one. 
        
        . First linear layer
        Linear(2 * memory_dim + msg_dim + 2, memory_dim)

        This takes the full event input and projects it down to a vector of size memory_dim.

        So if:

        memory_dim = 8
        msg_dim = 4

        then input size is:

        2(8)+4+2=22
        2(8)+4+2=22

        So the first layer is:

        Linear(22, 8)

        That means:

        input: 22 numbers
        output: 8 numbers

        This layer learns how to combine:

        -source state
        -destination state
        -message features
        -time gaps

           into one compact hidden representation.
        
        2. ReLU
        ReLU()

        This applies a non-linear activation.

        Very simply:

        positive values stay
        negative values become 0

        Why do we need it?

        Because without a non-linearity, the whole block would just behave 
        like one linear transformation.
        
        ReLU lets the model learn more flexible patterns.      
        
        . Second linear layer
        Linear(memory_dim, memory_dim)

        This takes the intermediate hidden vector and transforms it again.

        So if memory_dim = 8, this is:

        Linear(8, 8)

        This does not change the size, but it lets the model refine the event representation.

        So the final output is still an 8-dimensional vector, but now it is a learned event embedding.
        
        What is the purpose of this block? 
        Is getting something like: 
                
         “This is the model's learned summary of what just happened in this interaction.”

        That summary is then passed to the GRUCell to update the source and destination node memories.
        """
        self.memory_updater = GRUCell(
            input_size=memory_dim,
            hidden_size=memory_dim
        )

        """
        Final classifier.
        """
        self.classifier = Linear(memory_dim, num_classes) # This converts the final node representation into class logits.

    def forward(self, data):
        """
        Process the event stream in chronological order.
        """

        num_nodes = data.x.size(0)

        """
        Initial node memory.
        shape = [num_nodes, memory_dim]
        """
        memory = self.node_encoder(data.x)

        """
        Track the last time each node was updated.
        """
        last_update = torch.zeros(
            num_nodes,
            dtype=data.t.dtype,
            device=data.t.device
        )

        for i in range(data.src.size(0)):
            src = data.src[i].item()
            dst = data.dst[i].item()
            time = data.t[i]
            msg = data.msg[i]

            """
            Time since each node was last updated.
            """
            delta_src = time - last_update[src]
            delta_dst = time - last_update[dst]

            """
            Build one event representation.
            """
            event_input = torch.cat(
                [
                    memory[src],
                    memory[dst],
                    msg,
                    delta_src.view(1),
                    delta_dst.view(1),
                ],
                dim=0
            )

            event_embedding = self.event_mlp(event_input)

            """
            Update source memory.
            """
            new_src_memory = self.memory_updater(
                event_embedding.unsqueeze(0),
                memory[src].unsqueeze(0)
            ).squeeze(0)

            """
            Update destination memory.
            """
            new_dst_memory = self.memory_updater(
                event_embedding.unsqueeze(0),
                memory[dst].unsqueeze(0)
            ).squeeze(0)
            
            """
             Why unsqueeze(0)?

            Because GRUCell expects inputs with a batch dimension.

            It wants shapes like:

            [batch_size, feature_dim]

            But event_embedding and memory[src] are just 1D vectors, like:

            [feature_dim]

            So:

            unsqueeze(0)

            adds a batch dimension at the front.

            Example:

            before: shape [8]
            after: shape [1, 8]

            That makes the GRUCell happy.

            Then after the update, we do:

            .squeeze(0)

            to remove that temporary batch dimension again.

            So:

            input to GRUCell: [1, 8]
            output from GRUCell: [1, 8]
            after squeeze: [8]
            
            """

            """
            Clone for teaching clarity before assignment.
            """
            memory = memory.clone()
            memory[src] = new_src_memory
            memory[dst] = new_dst_memory

            """
            memory is the matrix that stores the current hidden state of all nodes.
            This block makes a fresh copy of the full node-memory matrix and 
            then replaces the rows for the source and destination nodes 
            with their newly updated hidden states.
            
            Its shape is:

            [num_nodes, memory_dim]

            So if you had 8 nodes and memory_dim = 8, then memory is an 8 x 8 tensor.

            Each row is one node's current hidden state.
            
            Then these two lines
            
            memory[src] = new_src_memory
            memory[dst] = new_dst_memory

            mean:

            replace the old memory of the source node with the new source memory
            replace the old memory of the destination node with the new destination memory

            So after this, the global memory matrix reflects the latest event.



            """
            last_update[src] = time
            last_update[dst] = time

        logits = self.classifier(memory)

        return logits


# ------------------------------------------------------------------
# PART 3: TRAINING FUNCTION
# ------------------------------------------------------------------

def train(model, data, optimizer):
    """
    Train the model on all node labels.
    """
    model.train()
    optimizer.zero_grad()

    logits = model(data)
    loss = F.cross_entropy(logits, data.y)

    loss.backward()
    optimizer.step()

    return loss.item()


# ------------------------------------------------------------------
# PART 4: EVALUATION FUNCTION
# ------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, data):
    """
    Evaluate node classification accuracy.
    """
    model.eval()

    logits = model(data)
    preds = logits.argmax(dim=1)

    correct = (preds == data.y).sum().item()
    total = data.y.size(0)

    accuracy = correct / total if total > 0 else 0.0

    return accuracy, preds


# ------------------------------------------------------------------
# PART 5: MAIN
# ------------------------------------------------------------------

if __name__ == "__main__":

    NODES_CSV = "nodes.csv"
    EVENTS_CSV = "events.csv"

    NODE_FEATURE_COLS = ["feature_1", "feature_2", "feature_3"]
    EVENT_FEATURE_COLS = ["event_weight", "channel_code", "priority", "response_time"]

    """
    Set label_col to None to derive toy labels from event participation count.
    Or set it to a real label column present in nodes.csv.
    Example: LABEL_COL = "label"
    """
    LABEL_COL = None

    data = load_continuous_time_event_graph_from_csv(
        nodes_csv_path=NODES_CSV,
        events_csv_path=EVENTS_CSV,
        node_id_col="node_id",
        src_col="src",
        dst_col="dst",
        time_col="timestamp",
        node_feature_cols=NODE_FEATURE_COLS,
        event_feature_cols=EVENT_FEATURE_COLS,
        label_col=LABEL_COL,
    )

    print("TemporalData object:")
    print(data)
    print()

    print("Node IDs:", data.node_ids)
    print("Number of nodes:", data.x.size(0))
    print("Number of events:", data.src.size(0))
    print("Node feature shape:", data.x.shape)
    print("Event message shape:", data.msg.shape)
    print("First 5 source nodes:", data.src[:5])
    print("First 5 destination nodes:", data.dst[:5])
    print("First 5 timestamps:", data.t[:5])
    print("Event counts per node:", data.event_counts)
    print("Node labels:", data.y)
    print()

    in_channels = data.x.size(1)
    msg_dim = data.msg.size(1)

    model = CTDGModel(
        in_channels=in_channels,
        msg_dim=msg_dim,
        memory_dim=8,
        num_classes=2
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("Training...\n")

    for epoch in range(1, 101):
        loss = train(model, data, optimizer)

        if epoch % 10 == 0:
            acc, _ = evaluate(model, data)
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")

    final_acc, final_preds = evaluate(model, data)

    print("\nFinal accuracy:", final_acc)
    print("Predicted labels:")
    print(final_preds)

    print("\nTrue labels:")
    print(data.y)