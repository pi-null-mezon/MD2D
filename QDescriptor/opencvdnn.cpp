/**M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace cv::dnn;

#include <fstream>
#include <iostream>
#include <cstdlib>
using namespace std;

#include <QDir>
#include <QByteArray>
#include <QStringList>
#include <QTextStream>
#include <QString>
#include <QFile>

std::vector<String> readClassNames(const char *filename = "C:/Programming/3rdParties/Caffe/models/vgg_face_caffe/names.txt");

cv::Mat describe(const String &filename, dnn::Net &net);
cv::Mat preprocessimageforVGGCNN(const Mat &_inmat);
cv::Mat cropresize(const cv::Mat &input, const cv::Size size);
void    recognize(const String &filename, dnn::Net &net);

int main(int argc, char **argv)
{
    setlocale(LC_CTYPE, "rus");

    String modelTxt = "C:/Programming/3rdParties/Caffe/models/vgg_face_caffe/VGG_FACE_deploy.prototxt";
    String modelBin = "C:/Programming/3rdParties/Caffe/models/vgg_face_caffe/VGG_FACE.caffemodel";
    //! [Initialize network]
    dnn::Net net = dnn::readNetFromCaffe(modelTxt,modelBin);
    if(net.empty()) {
        std::cerr << "Can not load neural network! Did you have download it? Abort...";
        return -2;
    }

    QFile _flbls("labels.txt");
    QFile _fdscr("descriptions.txt");
    if(_flbls.open(QFile::WriteOnly) == false || _fdscr.open(QFile::WriteOnly) == false) {
        qWarning("Can not open output files for writing! Abort...");
        return -2;
    }

    QDir _dir("C:/Testdata/Face/Test/diacare_singleshot");
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
            qInfo("%d.%d) %s", i, j, _filenames.at(j).toLocal8Bit().constData());
            cv::Mat _dm = describe(_subdir.absoluteFilePath(_filenames.at(j)).toLocal8Bit().constData(),net);
            float *_p = _dm.ptr<float>(0);
            for(size_t k = 0; k < _dm.total() - 1; ++k) {
                _fdscr.write(QString::number(_p[k],'f',10).append(", ").toUtf8());
            }
            _fdscr.write(QString::number(_p[_dm.total()-1],'f',10).append('\n').toUtf8());
            _fdscr.flush();

            _flbls.write(("" + _subdir.absoluteFilePath(_filenames.at(j)) + "\n").toUtf8());
            _flbls.flush();           
        }
    }
    return 0;
}

//------------------------------------------------------------------------

void recognize(const String &filename, dnn::Net &net)
{
    Mat img = imread(filename, CV_LOAD_IMAGE_UNCHANGED);
    if (img.empty())
    {
        std::cerr << "Can't read image from the file: " << filename << std::endl;
        exit(-1);
    }

    img = preprocessimageforVGGCNN(img);
    cv::imshow("Probe", img);
    cv::waitKey(1);

    //! [Prepare blob]
    dnn::Blob inputBlob = dnn::Blob::fromImages(img);   //Convert Mat to dnn::Blob image batch
    //! [Prepare blob]

    //! [Set input blob]
    net.setBlob(".data", inputBlob);
    //! [Set input blob]

    //! [Make forward pass]
    net.forward();
    //! [Make forward pass]

    //! [Gather output]
    dnn::Blob prob = net.getBlob("prob");   //gather output of "prob" layer

    Mat probMat = prob.matRefConst().reshape(1, 1); //reshape the blob to 1xN vector
    Point classNumber;

    double classProb;
    minMaxLoc(probMat, NULL, &classProb, NULL, &classNumber);
    int classId = classNumber.x;

    std::vector<String> classNames = readClassNames();
    std::cout << "Best class: #" << classId << " '" << classNames.at(classId) << "'" << std::endl;
    std::cout << "Probability: " << classProb * 100 << "%" << std::endl;
}

cv::Mat describe(const String &filename, dnn::Net &net)
{
    Mat img = imread(filename, CV_LOAD_IMAGE_UNCHANGED);
    if (img.empty())
    {
        std::cerr << "Can't read image from the file: " << filename << std::endl;
        exit(-1);
    }

    img = preprocessimageforVGGCNN(img);
    cv::imshow("Probe", img);
    cv::waitKey(1);

    //! [Prepare blob]
    dnn::Blob inputBlob = dnn::Blob::fromImages(img);
    //! [Prepare blob]

    //! [Set input blob]
    net.setBlob(".data", inputBlob);
    //! [Set input blob]

    //! [Make forward pass]
    net.forward();
    //! [Make forward pass]

    dnn::Blob featuresBlob = net.getBlob("fc8");
    return featuresBlob.matRefConst().reshape(1, 1);
}

std::vector<String> readClassNames(const char *filename)
{
    std::vector<String> classNames;

    std::ifstream fp(filename);
    if (!fp.is_open())
    {
        std::cerr << "File with classes labels not found: " << filename << std::endl;
        exit(-1);
    }

    std::string name;
    while (!fp.eof())
    {
        std::getline(fp, name);
        if (name.length())
            classNames.push_back( name.substr(name.find(' ')+1) );
    }

    fp.close();
    return classNames;
}

cv::Mat preprocessimageforVGGCNN(const Mat &_inmat)
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
