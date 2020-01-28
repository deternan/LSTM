package com.deeplearning4j.dl4j;

/*
 * Word2Vector
 * Java version
 * 
 * version: September 20, 2019 09:54 PM
 * Last revision: September 21, 2019 04:46 PM
 * 
 * Author : Chao-Hsuan Ke 
 * E-mail : phelpske.dev at gmail dot com
 * 
 */

/*
 * Reference
 * https://deeplearning4j.org/docs/latest/deeplearning4j-nlp-word2vec
 * 
 */

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;

import java.io.File;
import java.io.IOException;

public class Word2Vector
{
	private String modelPath = "";						// path
	private String modelgzName = "GoogleNews-vectors-negative300.bin.gz";		// Google News (English)
	private String modeltxtName = "zh_wiki_word2vec_300.txt";					// Wiki		   (Chinese)
	
	File gModel;
	WordVectors word2Vec;
	//Word2Vec
	
//	String inputStr = "西元前";
	String inputStr = "apple";
	double[] wordVectorDouble = {};	
	
	public Word2Vector()
	{
		ReadBin();
		//ReadTXT();
	  
        // list word dim value
        //getWordDim();
        
        // Compared with two word similarity
        WordSimilar("apple", "iPhone");
	}
	
	private void ReadBin() {
		gModel = new File(modelPath + modelgzName);
	    word2Vec = WordVectorSerializer.readWord2VecModel(gModel);
	}
	
	private void ReadTXT() {
		gModel = new File(modelPath + modeltxtName);
		try {
			word2Vec = WordVectorSerializer.loadTxtVectors(gModel);			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	private void getWordDim() {
        
		// list word dim value
		System.out.println("inputStr	"+inputStr);
		
		if(word2Vec.hasWord(inputStr)) {
			wordVectorDouble = word2Vec.getWordVector(inputStr);
			//System.out.println("dim length	"+wordVectorDouble.length);
			for(int i=0; i<wordVectorDouble.length; i++) {
				System.out.println(wordVectorDouble[i]);
			}
		}else {
			System.out.println("no term in the model");
		}	
		
	}
	
	private void WordSimilar(String str1, String str2) {

		double cosSimChi = word2Vec.similarity(str1, str2);
		System.out.println(cosSimChi);
	}
	
	public static void main(String[] args) 
	{
		Word2Vector w2c = new Word2Vector();
	}
	
}

