#include <QStringList>
#include <QTextStream>
#include <QFile>
#include <QDir>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#include <dlib/dnn.h>

#define CHECK_DLIB_FACEDESCRIPTOR

// ----------------------------------------------------------------------------------------
namespace dlib {
    template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
    using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

    template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
    using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

    template <int N, template <typename> class BN, int stride, typename SUBNET>
    using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

    template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
    template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

    template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
    template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
    template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
    template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
    template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

    using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                                alevel0<
                                alevel1<
                                alevel2<
                                alevel3<
                                alevel4<
                                max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                                input_rgb_image_sized<150>
                                >>>>>>>>>>>>;
}
//----------------------------------------------------------------------------------------

dlib::frontal_face_detector dlibfacedet;
dlib::shape_predictor       dlibshapepredictor;

cv::Mat                         getVGGDescription(const cv::String &filename, cv::dnn::Net &net);
cv::Mat                         getDlibDescription(const cv::String &filename, dlib::anet_type &net);
cv::Mat                         prepareFaceImageForVGG(const cv::Mat &_inmat);
dlib::matrix<dlib::rgb_pixel>   prepareFaceImageForDlib(const cv::Mat &_inmat);
cv::Mat                         cropresize(const cv::Mat &input, const cv::Size size);

int main(int argc, char **argv)
{
    setlocale(LC_CTYPE, "rus");

#ifndef CHECK_DLIB_FACEDESCRIPTOR
    cv::String modelTxt = "C:/Programming/3rdParties/Caffe/models/vgg_face_caffe/VGG_FACE_deploy.prototxt";
    cv::String modelBin = "C:/Programming/3rdParties/Caffe/models/vgg_face_caffe/VGG_FACE.caffemodel";
    cv::dnn::Net net = cv::dnn::readNetFromCaffe(modelTxt,modelBin);
    if(net.empty()) {
        std::cerr << "Can not load neural network! Did you have download it? Abort...";
        return -2;
    }
#else
    dlib::anet_type dlibnet;
    try {
        dlibfacedet = dlib::get_frontal_face_detector();
        dlib::deserialize("C:/Programming/3rdParties/Dlib/build/etc/data/shape_predictor_68_face_landmarks.dat") >> dlibshapepredictor;
        dlib::deserialize("C:/Programming/3rdParties/Dlib/build/etc/data/dlib_face_recognition_resnet_model_v1.dat") >> dlibnet;
    }
    catch (const cv::Exception &err) {
        std::cerr << err.msg << std::endl;
    }
#endif

    QFile _flbls("labels.txt");
    QFile _fdscr("descriptions.txt");
    if(_flbls.open(QFile::WriteOnly) == false || _fdscr.open(QFile::WriteOnly) == false) {
        qWarning("Can not open output files for writing! Abort...");
        return -2;
    }

    QDir _dir;
    if(argc > 1) {
        _dir = QDir(argv[1]);
    } else {
        qWarning("Please provide source directory to process! Abort...");
        return -1;
    }
    if(!_dir.exists()) {
        qWarning("Can not find directory with files! Abort...");
        return -1;
    }

    QStringList _subdirnames = _dir.entryList(QDir::NoDotAndDotDot | QDir::Dirs);
    for(int i = 0; i < _subdirnames.size(); ++i) {
        QString _subdirname = _subdirnames.at(i);
        qInfo("%d) %s", i, _subdirname.toUtf8().constData());

        QDir _subdir(_dir.absolutePath().append("/").append(_subdirnames.at(i)));
        QStringList _filenames = _subdir.entryList(QDir::NoDotAndDotDot | QDir::Files);
        for(int j = 0; j < _filenames.size(); ++j) {
            qInfo("   %d.%d) %s", i, j, _filenames.at(j).toUtf8().constData());
#ifndef CHECK_DLIB_FACEDESCRIPTOR
            cv::Mat _dm = getVGGDescription(_subdir.absoluteFilePath(_filenames.at(j)).toLocal8Bit().constData(),net);
#else
            cv::Mat _dm = getDlibDescription(_subdir.absoluteFilePath(_filenames.at(j)).toLocal8Bit().constData(),dlibnet);
#endif
            if(!_dm.empty()) {
                float *_p = _dm.ptr<float>(0);
                for(size_t k = 0; k < _dm.total() - 1; ++k) {
                    _fdscr.write(QString::number(_p[k],'f',7).append(", ").toUtf8());
                }
                _fdscr.write(QString::number(_p[_dm.total()-1],'f',7).append('\n').toUtf8());
                _fdscr.flush();

                _flbls.write(("" + _subdir.absoluteFilePath(_filenames.at(j)) + "\n").toUtf8());
                _flbls.flush();
            }
        }
    }
    return 0;
}

//------------------------------------------------------------------------

cv::Mat getVGGDescription(const cv::String &filename, cv::dnn::Net &net)
{
    cv::Mat img = imread(filename, CV_LOAD_IMAGE_UNCHANGED);
    if(img.empty()) {
        std::cerr << "Can't read image from the file: " << filename
                  << ". It will be excluded from the analysis!" << std::endl;
        return cv::Mat();
    }
    img = prepareFaceImageForVGG(img);
    cv::imshow("Probe", img);
    cv::waitKey(1);
    cv::Mat inputBlob = cv::dnn::blobFromImage(img);
    net.setInput(inputBlob, "data");
    return net.forward("pool5").reshape(1,1);
}

cv::Mat prepareFaceImageForVGG(const cv::Mat &_inmat)
{
    // VGG Face CNN accepts only 224 x 224 RGB-images
    cv::Mat _outmat;

    if(_inmat.channels() == 1) {
        cv::cvtColor(_inmat,_outmat,CV_GRAY2RGB);
    } else if(_inmat.channels() == 4) {
        cv::cvtColor(_inmat,_outmat,CV_BGRA2RGB);
    } else if(_inmat.channels() == 3) {
        cv::cvtColor(_inmat,_outmat,CV_BGR2RGB);
    } else {
        _outmat = _inmat;
    }

    if(_outmat.cols != 224 || _outmat.rows != 224) {
        _outmat = cropresize(_outmat, cv::Size(224,224));
    }
    return _outmat;
}

dlib::matrix<dlib::rgb_pixel> prepareFaceImageForDlib(const cv::Mat &_inmat)
{
    cv::Mat _rgbmat;
    cv::Mat _graymat;
    if(_inmat.channels() == 1) {
        _graymat = _inmat;
        cv::cvtColor(_inmat,_rgbmat, CV_GRAY2BGR);
    } else {
        cv::cvtColor(_inmat, _rgbmat, CV_BGR2RGB);
        cv::cvtColor(_inmat, _graymat, CV_BGR2GRAY);
    }

    dlib::cv_image<unsigned char> _graycv_image(_graymat);
    dlib::cv_image<dlib::rgb_pixel> _rgbcv_image(_rgbmat);

    dlib::rectangle _facerect(_inmat.cols,_inmat.rows);
    std::vector<dlib::rectangle> _facerects = dlibfacedet(_graycv_image);
    if(_facerects.size() > 0) {
        _facerect = _facerects[0];
    }
    auto _shape = dlibshapepredictor(_graycv_image, _facerect);
    dlib::matrix<dlib::rgb_pixel> face_chip;
    dlib::extract_image_chip(_rgbcv_image, dlib::get_face_chip_details(_shape,150,0.25), face_chip);
    return face_chip;
}

cv::Mat getDlibDescription(const cv::String &filename, dlib::anet_type &net)
{
    cv::Mat img = imread(filename, CV_LOAD_IMAGE_UNCHANGED);
    if (img.empty()) {
        std::cerr << "Can't read image from the file: " << filename
                  << ". It will be excluded from the analysis!" << std::endl;
        return cv::Mat();
    }
    dlib::matrix<dlib::rgb_pixel> face_chip = prepareFaceImageForDlib(img);
    cv::Mat _viewmat = dlib::toMat(face_chip);
    cv::imshow("Input of DLIB",_viewmat);
    cv::waitKey(1);
    dlib::matrix<float,0,1> _facedescription = net(face_chip);
    return dlib::toMat(_facedescription).reshape(1,1).clone();
}

cv::Mat cropresize(const cv::Mat &input, const cv::Size size)
{
    cv::Rect2f roiRect(0,0,0,0);
    if( (float)input.cols/input.rows > (float)size.width/size.height) {
        roiRect.height = (float)input.rows;
        roiRect.width = input.rows * (float)size.width/size.height;
        roiRect.x = (input.cols - roiRect.width)/2.0f;
    } else {
        roiRect.width = (float)input.cols;
        roiRect.height = input.cols * (float)size.height/size.width;
        roiRect.y = (input.rows - roiRect.height)/2.0f;
    }
    roiRect &= cv::Rect2f(0, 0, (float)input.cols, (float)input.rows);
    cv::Mat output;
    if(roiRect.area() > 0)  {
        cv::Mat croppedImg(input, roiRect);
        int interpolationMethod = 0;
        if(size.area() > roiRect.area())
            interpolationMethod = CV_INTER_CUBIC;
        else
            interpolationMethod = CV_INTER_AREA;
        cv::resize(croppedImg, output, size, 0, 0, interpolationMethod);
    }
    return output;
}
