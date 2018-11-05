#include "FaceDescriptorExtractor.h"


FaceDescriptorExtractor::FaceDescriptorExtractor()//Nuestro constructor por default, el cuál se manda a llamar en caso de que no reciba un path y lo carga una sola vez
{
    deserialize("../faces/dlib_face_recognition_resnet_model_v1.dat") >> net;//Cargamos nuestro modelo y declaramos nuestra net
}

FaceDescriptorExtractor::FaceDescriptorExtractor(std::string path)//Nuestro constructor, el cuál recibe el path del modelo entrenado y lo carga una sola vez
{
    deserialize(path) >> net;//Cargamos nuestro modelo y declaramos nuestra net
}

FaceDescriptorExtractor::~FaceDescriptorExtractor() {}//Nuestro destructor

cv::Mat FaceDescriptorExtractor::obtenerDescriptorVectorial(cv::Mat &rostro)//Nuestra función para obtener la descripción vectorial del rostro de una persona
{
    dlib::cv_image<bgr_pixel> image(rostro);//Primero transformamos el Mat que recibimos a una matrix de rgb_pixel
    dlib::matrix<rgb_pixel> matriz;
    dlib::assign_image(matriz, image);
    std::vector<matrix<rgb_pixel>> face;// Creamos un vector de matrices donde guardamos el rostro que identificamos
    face.push_back(matriz);//Lo agregamos
    
    std::vector<matrix<float,0,1>> face_descriptors = net(face);//Aquí obtenemos el desciptor vectorial del rostro que cargamos 
    cv::Mat descriptor;
    descriptor = toMat(face_descriptors[0]); //Guarda en descriptor la dirreccion en memoria del face_descriptor
    return descriptor.clone();//Lo regresamos clonado para que devuelva los valores correctos en el MAT y no solo la direccion
}

//En esta función recibimos dos MAT la primera siendo la imagen captada en tiempo real y la segundo la que está guardada en la base de datos
bool FaceDescriptorExtractor::compararDescriptores(cv::Mat &rostroReal, cv::Mat &rostroBD)
{
    double distanciaEuclidiana = 0;
    distanciaEuclidiana = cv::norm(rostroReal, rostroBD, cv::NORM_L2);
    if (distanciaEuclidiana < 0.6)//Calculamos la distancia euclidiana y si es menor a 0.6, que es un umbral que definimos, afirmamos que es la misma 
    {
        return true;
    }
    return false;
}
