from .tree_log import *
from .tree_snapshot import *
from .visualizer_client import VisualizerClient
from ..algorithm import MCTSResult


class ReasonersVisualizer:
    @staticmethod
    def visualize(result: Union[TreeLog, MCTSResult], **kwargs):

        tree_log: TreeLog

        if isinstance(result, TreeLog):
            tree_log = result
        elif isinstance(result, MCTSResult):
            tree_log = TreeLog.from_mcts_results(result, **kwargs)
        elif isinstance(result, ...):
            raise NotImplementedError()
        else:
            raise TypeError(f"Unsupported result type: {type(result)}")

        receipt = VisualizerClient().post_log(tree_log)

        if receipt is None:
            return

        return ReasonersVisualizer._present_visualizer(receipt)

    @staticmethod
    def _present_visualizer(receipt: VisualizerClient.TreeLogReceipt):

        import webbrowser
        print(f"Visualizer URL: {receipt.access_url()}")

        webbrowser.open(receipt.access_url())
