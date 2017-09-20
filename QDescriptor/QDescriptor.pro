QT += core
QT -= gui

CONFIG += c++11

TARGET = QDescriptor
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp

include(opencv.pri)
include(dlib.pri)
