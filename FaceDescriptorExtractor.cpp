#include "FaceDescriptorExtractor.h"

FaceDescriptorExtractor::FaceDescriptorExtractor(std::string path)//Nuestro constructor, el cuál recibe el path del modelo entrenado y lo carga una sola vez
{
    deserialize(path) >> net;//Cargamos nuestro modelo y declaramos nuestra net
}

FaceDescriptorExtractor::~FaceDescriptorExtractor() {}//Nuestro destructor

matrix<float,0,1> FaceDescriptorExtractor::obtenerDescriptorVectorial(cv::Mat &rostro)//Nuestra función para obtener la descripción vectorial del rostro de una persona
{
    dlib::cv_image<bgr_pixel> image(rostro);//Primero transformamos el Mat que recibimos a una matrix de rgb_pixel
    dlib::matrix<rgb_pixel> matriz;
    dlib::assign_image(matriz, image);
    std::vector<matrix<rgb_pixel>> face;// Creamos un vector de matrices donde guardamos el rostro que identificamos
    face.push_back(matriz);//Lo agregamos
    std::vector<matrix<float,0,1>> face_descriptors = net(face);//Aquí obtenemos el desciptor vectorial del rostro que cargamos 
    return face_descriptors[0];//Lo regresamos
}

//En esta función recibimos dos matrix<float,0,1> la primera siendo la imagen captada en tiempo real y la segundo la que está guardada en la base de datos
bool FaceDescriptorExtractor::compararDescriptores(dlib::matrix<float,0,1> &rostroReal, dlib::matrix<float,0,1> &rostroBD)
{
    if (length(rostroReal-rostroBD < 0.6))//Calculamos la distancia euclidiana y si es menor a 0.6, que es un umbral que definimos, afirmamos que es la misma 
    {
        return true;
    }
    //std::cout << "Distancia euclediana= " << length(rostroReal-rostroBD) << std::endl;
    return false;
}
