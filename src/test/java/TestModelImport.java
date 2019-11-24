import static org.junit.jupiter.api.Assertions.assertEquals;


import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;


class TestModelImport {
    @Test
    void testImport() throws Exception {
        KerasLayer.registerCustomLayer("SumRows", KerasRowSumLayer.class);

        InputStream modelIn = getClass().getResourceAsStream("/model.h5");
        InputStream expectedLayerIn = getClass().getResourceAsStream("result.dat");
        InputStream inputIn = getClass().getResourceAsStream("input.dat");
        INDArray expectedResult = Nd4j.createNpyFromInputStream(expectedLayerIn);
        INDArray input  = Nd4j.createNpyFromInputStream(inputIn);


        ComputationGraph model = KerasModelImport.importKerasModelAndWeights(modelIn);

        INDArray[] outputs = model.output(input);
        assertEquals(outputs.length, 1);
        assertEquals(outputs[0], expectedResult); //really this should be approx equal
    }
}
