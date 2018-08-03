#include <stdio.h>
#include <stdlib.h>
//#include <opencv/highgui.h> 
//#include <opencv/cv.h> 
#include <opencv2/opencv.hpp>
#include "Python.h"
#include <numpy/arrayobject.h>
using namespace cv;
int flag = 0;
#define  SAVE_VIDEO 
#ifdef SAVE_VIDEO
cv::Mat img;
# define  SAVE_FRAMES  500
cv::VideoWriter writer("VideoResult.avi", CV_FOURCC('X', 'V', 'I', 'D'), 15, Size(640, 480)); 
int frame_num = 0;
#endif
void init_numpy(){

	import_array();
}
PyObject *pName,*pModule,*pFunc;
void py_init()
{
	Py_Initialize();
	if ( !Py_IsInitialized() ) {
		printf("init err\n");
	}else{
		printf("init ok\n");
	}
	printf("finding ...\n");
	init_numpy();
	/*pName = PyString_FromString("Test111");


	if(!pName){
		printf("finding err \n");
	}else{
		printf("finding ok \n");
	}*/
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append('./')");
	pModule = PyImport_ImportModule("mvnc-mobile-ssd");
	if ( !pModule ) {
		printf("can't find .py");
	}else{
		printf("py found\n");
	}
	if(flag == 0)
	{
		pFunc = PyObject_GetAttrString(pModule, "init");
		PyObject_CallObject(pFunc, NULL);////初始化计算棒
		pFunc = PyObject_GetAttrString(pModule, "process2");
	}
	flag =1;

}
void py_free()
{
	//pFunc = PyObject_GetAttrString(pModule, "release");
	//PyObject_CallObject(pFunc, NULL);
	if(pName)
		Py_DECREF(pName);
	if(pModule)
		Py_DECREF(pModule);
	if(pFunc)
		Py_DECREF(pFunc);
	// 关闭Python
	Py_Finalize();
}
int main()
{
	py_init();
	VideoCapture cap("video.avi");
	{
		if(!cap.isOpened()) // check if we succeeded

			return -1;

		while(1)
		{
			Mat frame;
			cap >> frame;
			if(frame.empty())
				break;
#ifdef SAVE_VIDEO
			frame.copyTo(img);
#endif
			unsigned char* imagedata = (unsigned char *)malloc(frame.cols * frame.rows * 3);
			memcpy(imagedata, frame.data, frame.cols * frame.rows * 3);
			npy_intp Dims[2]= { frame.rows, frame.cols * 3}; //给定维度信息
			PyObject* PyListRGB = PyArray_SimpleNewFromData(2, Dims, NPY_UBYTE, imagedata);
			//int Index_i = 0;
			//PyObject *PyListRGB  = PyList_New(frame.cols * frame.rows * 3);//定义一个与数组等长的PyList对象数组
			//for(Index_i = 0; Index_i < PyList_Size(PyListRGB); Index_i++){
			//
			//	PyList_SetItem(PyListRGB, Index_i,  Py_BuildValue("i", imagedata[Index_i]));//给PyList对象的每个元素赋值
			//}
			PyObject *ArgList = PyTuple_New(3);
			PyTuple_SetItem(ArgList, 0, PyListRGB);//将PyList对象放入PyTuple对象中
			PyTuple_SetItem(ArgList, 1, Py_BuildValue("i", frame.cols));
			PyTuple_SetItem(ArgList, 2, Py_BuildValue("i", frame.rows));

			//printf("start \n");
			/*PyObject_CallObject(pFunc, ArgList);//调用函数，完成传递
			//printf("end\n");fflush(NULL);
			free(imagedata);
			//if(PyListRGB)
			//	Py_DECREF(PyListRGB);
			//printf("end1\n");fflush(NULL);
			if(ArgList)
				Py_DECREF(ArgList);*/
			PyObject* Pyresult= PyObject_CallObject(pFunc, ArgList);//调用函数，完成传递
			PyObject* ret_objs;
			PyArg_Parse(Pyresult, "O!", &PyList_Type, &ret_objs);
			int size = PyList_Size(ret_objs);
			int i,j;
			int nboxes = 0;
			int rst[600]={0};
			for(i=0;i<size/6;i++)
			{

				int class_id, confidence, x, y, w, h;
				class_id = PyInt_AsLong(PyList_GetItem(ret_objs,i*6+0));
				confidence = PyInt_AsLong(PyList_GetItem(ret_objs,i*6+1));
				x=PyInt_AsLong(PyList_GetItem(ret_objs,i*6+2));
				y=PyInt_AsLong(PyList_GetItem(ret_objs,i*6+3));
				w=PyInt_AsLong(PyList_GetItem(ret_objs,i*6+4));
				h=PyInt_AsLong(PyList_GetItem(ret_objs,i*6+5));

				rst[i * 6 + 0] = class_id;
				rst[i * 6 + 1] = confidence;
				rst[i * 6 + 2] = x;
				rst[i * 6 + 3] = y;
				rst[i * 6 + 4] = w;
				rst[i * 6 + 5] = h;
#ifdef SAVE_VIDEO
				cv::rectangle(img, cv::Rect(x,y,w,h), cv::Scalar(255, 0 ,0), 1, 8, 0 );
#endif
				nboxes++;
			}
			printf("detect num = %d\n",nboxes);
			free(imagedata);
			if(Pyresult)
				Py_DECREF(Pyresult);
			//if(PyListRGB)
			//	Py_DECREF(PyListRGB);
			if(ArgList)
				Py_DECREF(ArgList);
#ifdef SAVE_VIDEO
			writer << img; 
			if(frame_num > SAVE_FRAMES)
				writer.release();
			frame_num++;
#endif 

		}
		py_free();
	}

	return 1;
}