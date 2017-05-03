/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package semana07;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;
import weka.classifiers.lazy.IBk;
import weka.core.Debug;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Guilherme Agunzo
 */
public class IrisKnn {
    
    public static void main(String[] args) throws FileNotFoundException, IOException, Exception {
        
        // DEFININDO CONJUNTO DE TREINAMENTO
        
        // - Definindo o leitor do arquivo arff
        
        FileReader baseIris = new FileReader("iris.arff");
        // - Definindo o grupo de instancias a partir do arquivo "simpsons.arff"
        
        Instances iris = new Instances(baseIris);
        
        // - Definindo o indice do atributo classe
        
        iris.setClassIndex(4);
        
        iris = iris.resample(new Debug.Random());
        
        Instances irisTreino = iris.trainCV(3, 0);
        Instances irisTeste = iris.testCV(3, 0);
        
        // DEFININDO EXEMPLO DESCONHECIDO
        
        //5.9,3.0,5.1,1.8,Iris-virginica
        Instance irisInst = new DenseInstance(iris.numAttributes());
        irisInst.setDataset(iris);
        irisInst.setValue(0, 5.9);
        irisInst.setValue(1, 3.0);
        irisInst.setValue(2, 5.1);
        irisInst.setValue(3, 1.8);
        
        // DEFININDO ALGORITMO DE CLASSIFICAÇÃO
        
        //NN
        
        IBk vizinhoIris = new IBk();
        
        //kNN
        
        IBk knnIris = new IBk(3);
        
        // MONTANDO CLASSIFICADOR
        //NN
        
        vizinhoIris.buildClassifier(irisTreino);
        
        //kNN
        
        knnIris.buildClassifier(irisTreino);
        
        // Definindo arquivo a ser escrito
        FileWriter writer = new FileWriter("iris.csv");
        
        // Escrevendo o cabeçalho do arquivo
        writer.append("Classe Real;Resultado NN;Resultado kNN");
        writer.append(System.lineSeparator());
        
        // Saída CLI / Console
        System.out.println("Classe Real;Resultado NN;Resultado kNN"); //Cabeçalho
        for(int i=0;i <= irisTeste.numInstances()-1;i++){
        
            Instance testeIris = irisTeste.instance(i);
            
            // Saída CLI / Console do valor original
            System.out.print(testeIris.stringValue(4)+";");
            
            // Escrevendo o valor original no arquivo
            writer.append(testeIris.stringValue(4)+";");
            
            // Definindo o atributo classe como indefinido
            testeIris.setClassMissing();

            // CLASSIFICANDO A INSTANCIA
            // NN
            
            double respostaVizinho = vizinhoIris.classifyInstance(testeIris);
            testeIris.setValue(4, respostaVizinho);
            String stringVizinho = testeIris.stringValue(4);
            
            //kNN

            double respostaKnn = knnIris.classifyInstance(testeIris);
            
            // Atribuindo respota ao valor do atributo do index 4(classe)
                
            testeIris.setValue(4, respostaKnn);
            
            String stringKnn = testeIris.stringValue(4);
            // Adicionando resultado ao grupo de instancia iris
            
            iris.add(irisInst);

            //Escrevendo os resultados no arquivo iris.csv
            
            writer.append(stringVizinho + ";");
            writer.append(stringKnn + ";");
            writer.append(System.lineSeparator());
            
            // Exibindo via CLI / Console o resultado
            
            System.out.print(respostaVizinho+";");
            System.out.print(respostaKnn+";");
            System.out.println(testeIris.stringValue(4));
        }
        
        writer.flush();
        writer.close();
        
    }
    
}
