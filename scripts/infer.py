import torch
import json


class InferenceEngine:
    """
    <Purpose>

    Attributes
    ----------
    atr1 : dtype
        <purpose>

    Methods
    -------
    __init__():
        Constructor
    m(p):
        <purpose>
    
    """

    device = None
    
    def __init__(self, configFile):
        """
        <Purpose>

        Parameter
        ---------
        attr1 : dtype
            <purpose>

        Raises
        ------
        err1
            <purpose>

        Returns
        -------
        tuple
            This function returns a tuple of (PPI_GDA_Data object, datagraph obj)

        """
        super(InferenceEngine, self).__init__()

        with open(configFile, "r") as cfg:
            self.config = json.load(cfg)
        InferenceEngine.device = self.config["hw"] if torch.cuda.is_available() else 'cpu'

        self.lossfn = None

    def infer(self, model, data, labels, test_mask):
        model.eval()
        
        logits = model(data)
        output = logits.argmax(1)

        return logits

