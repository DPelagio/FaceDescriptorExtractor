#include "FaceDescriptorExtractor.h"
#include <iostream>
#include <stdlib.h>

using namespace std;
using namespace cv;

int main()
    {
      //En esta demo, nosotros cargamos una imagen utilizando opencv como los demás módulos  
      cv::Mat image = imread("../johns/a.jpg", CV_LOAD_IMAGE_COLOR);//la cargamos a color
      cv::Mat image2 = imread("../johns/b.jpg", CV_LOAD_IMAGE_COLOR);//cargamos dos imágenes para demostrar cómo se utilizan todas nuestras funciones
      cv::imshow("Imagen Dada", image);//Aquí dejamos las imágenes en pantalla para demostración visual
      cv::imshow("Imagen Dada 2", image2);
      //Nosotros instanciamos nuestra clase creando un objeto con el nombre de nuestra clase y dándole un string del path donde se encuentra nuestro modelo entrenado de reconocimiento visual
      FaceDescriptorExtractor fr("../faces/dlib_face_recognition_resnet_model_v1.dat");
      //Aquí creamos una matriz de la librería dlib, ya que esto es lo que devuelve nuestra función obtenerDescripcionVectorial
      Mat vector1 = fr.obtenerDescriptorVectorial(image);//Nuestra función obtenerDescripcionVectorial recibe un Mat a color de una imagen de 150x150
      Mat vector2 = fr.obtenerDescriptorVectorial(image2);//Nuestra función regresa una matrix<float,0,1> la cual pertenece a dlib
      //Nuestra segunda función recibe 2 matrix<float,0,1> y regresa true si es que la persona es la misma, y regresa false si no lo es
      
      if(fr.compararDescriptores(vector1, vector2)){//Aquí simplemente hacemos una demostración de que si son la misma persona o no
          std::cout << "PERMITIDO" << std::endl;
      }
      else 
        std::cout << "ACCESO DENEGADO" << std::endl;
      waitKey(0);//Este waitKey es para que las imágenes permanezcan en pantalla
    }
