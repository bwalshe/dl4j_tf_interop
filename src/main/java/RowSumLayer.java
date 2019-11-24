import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.samediff.SDLayerParams;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayer;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;

public class RowSumLayer extends SameDiffLayer {

    public RowSumLayer(){
        //keeps jackson happy
    }

    @Override
    public void defineParameters(SDLayerParams params) {}

    @Override
    public void initializeParameters(Map<String, INDArray> params) {}

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.FF) {
            throw new IllegalStateException("Invalid input row sum layer (layer name=\"" + getLayerName()
                    + "\"): Expected FF input, got " + inputType);
        }
        InputType.InputTypeFeedForward rnnType = (InputType.InputTypeFeedForward) inputType;
        long size = rnnType.getSize() + 1;
        return InputType.feedForward(size);
    }

    @Override
    public SDVariable defineLayer(SameDiff sd, SDVariable layerInput, Map<String, SDVariable> paramTable, SDVariable mask) {
        SDVariable rowSums = sd.sum(layerInput, true, 1);
        return sd.concat(1, layerInput, rowSums);
    }

}
