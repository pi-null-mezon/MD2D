QT += core
QT -= gui

CONFIG += c++11

TARGET = Sources
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += opencvdnn.cpp

include(opencv.pri)
