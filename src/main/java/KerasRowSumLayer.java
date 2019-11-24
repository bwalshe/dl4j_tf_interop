import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;

import java.util.Map;

public class KerasRowSumLayer extends KerasLayer {

    public KerasRowSumLayer(Map<String, Object> layerConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException{
        this(layerConfig, true);
    }

    public KerasRowSumLayer(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);
        this.layer = new RowSumLayer();
    }

    public RowSumLayer rowSumLayer(){
        return (RowSumLayer) this.layer;
    }

    @Override
    public InputType getOutputType(InputType... input){
        return rowSumLayer().getOutputType(-1, input[0]);
    }
}
