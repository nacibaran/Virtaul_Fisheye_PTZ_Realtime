#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <stdio.h>
#include <vector>
#include <array>
#include <cmath>
#include <cassert>
#include<ctime>
#include <iostream>
#include <cmath>
using namespace std;

#define NUMOF_STOREABLE_MAPS 3


/*
* 
* 
* 
* 
		'''	Beni Oku '''


	1- Kamera görüntüsü istenile þekilde gelmez ise düzeltme seçenekleri.

		- Resize ayarlarý yeniden yapýlandýrýlabilir. 
		- Kamera 180 derece fakat küresel görüntü veriyor olabilir. ( Bize küresel vermeyen düzlemsel bir görüntü lazým )
		- Kameradan okunan weight ve height deðerleri kodda manuel olarak verildi en uygun deðerler bu hesaplandýðý için
		  bu deðerleri fonksiyon parametresi ile göndererek otomatik her kameraya uygun hale getirilebilir ( Manuel hesap = daha düzgün görüntü ) 
		  
		  
	2- Görüntüde kasmalar - gecikmeler olursa izlenebilecek adýmlar. 
		- Resize deðerleri pixel oraný daha küçük bir deðer ile deðiþtirilebilir.
		- Multi threading kullanýlarak çift zamanlý koþma yapýp iþi paylaþtýrabiliriz ( Gerek olmayabilir. ) 
		- En mantýklýsý ise sað sola dönüþ açýlarýna göre ( Default 30 olarak beta deðerini ayarladým )
		  kamera matrisleri bir txt. dosyasýna yazýlýr. Ve her kare geldiðinde hesap yapmak yerine bu dosyalardan python numpy gibi bir c++ kütüphanesi ile okunabilir.
		- Matris deðerlerini hesaplayan kod uygulama dosyalarýnýn içine eklenmiþtir. Main.py programýný koþturmadan önce kamera görüntüsünün düzgün olduðunu teyit edip txt dosyalarýný yazdýrabilirsiniz.
	
	3- PTZ dönüþ hýzý 
		- Beta thetha alpha ve zoom deðiþkenleri dinamik olursa buton clicklerine göre hýz ayarlamasý yapýlabilir.
		- !! Kullaným esnasýnda dönüþ hýzý deðiþimi . 



	4- Acrop fotoðraf çekme
		- Acrop fotoðraf çekme iþlemi veri tabanýnda daha önce kaç adet fotoðraf çekildiðini tutmalý ( txt , MySql vs.vs ) 
		  bu deðere göre fotaðraflarý sýralayýp kaydetmeli aksi taktirde 1. fotoðraf sürekli güncellenerek 10 fotoðraf çekilse bile sadece en sondaki fotoðraf hafýzada kalacaktýr. 

		  
	
		  
		  
		  
		  *
		  *
		  *
		  */




class FishEyeWindow {
private:
	unsigned short int srcW_, srcH_, destW_, destH_;  // Signed de yapýlabilir. Karýþýklýk çýkmýyacaðýný düþünüyorum . 

	float al_, be_, th_, R_, zoom_;
	vector<cv::Mat*> mapXs_, mapYs_;
public:

	FishEyeWindow(int srcW, int srcH, int destW, int destH)
		: srcW_(srcW), srcH_(srcH), destW_(destW), destH_(destH),
		al_(0), be_(0), th_(0), R_(srcW / 2.0), zoom_(0.65),
		mapXs_(NUMOF_STOREABLE_MAPS, NULL), mapYs_(NUMOF_STOREABLE_MAPS, NULL) {}

	~FishEyeWindow() {
		array<vector<cv::Mat*>*, 2> maps = { &mapXs_, &mapYs_ };
		for (int i = 0; i < maps.size(); i++) {
			for (vector<cv::Mat*>::iterator it = maps[i]->begin(); it != maps[i]->end(); it++) {
				delete* it;
			}
		}
	}

	void buildMap(float alpha, float beta, float theta, float zoom, int idx = 0) {


		clock_t baslangic = clock(), bitis;


		assert(0 <= idx && idx < NUMOF_STOREABLE_MAPS);

		// Bu iþlem daha global bir katmanda tek seferlik yapýlabilir. 
		cv::Mat* mapX = new cv::Mat(destH_, destW_, CV_32FC1);
		cv::Mat* mapY = new cv::Mat(destH_, destW_, CV_32FC1);
		// Buildmap fonksiyonuna gelen parametreleri hesaplamak için atamalarýmýzý yapalým . 
		al_ = alpha;
		be_ = beta;
		th_ = theta;
		zoom_ = zoom;
		float al = al_ / 180.0f;
		float be = be_ / 180.0f;
		float th = th_ / 180.0f;


		float	A =		cosf(th) * cosf(al)		 -		 sinf(th) * sinf(al) * cosf(be);
		float	B =		sinf(th) * cosf(al)		 +		 cosf(th) * sinf(al) * cosf(be);
		float	C =		cosf(th) * sinf(al)		 +		 sinf(th) * cosf(al) * cosf(be);
		float	D =		sinf(th) * sinf(al)		 -		 cosf(th) * cosf(al) * cosf(be);



		float mR = zoom_ * R_;
		float mR2 = mR * mR;
		float mRsinBesinAl = mR * sin(be) * sin(al);
		float mRsinBecosAl = mR * sin(be) * cos(al);
		int centerV = int(destH_ / 2.0);
		int centerU = int(destW_ / 2.0);
		float centerY = srcH_ / 2.0;
		float centerX = srcW_ / 2.0;
		// # Matris deðerlerini hesaplayýp bir Matrise kaydetme iþlemi yapýyoruz.
		for (int absV = 0; absV < destH_; absV++) {
			float v = absV - centerV;
			float vv = v * v;
			for (int absU = 0; absU < destW_; absU++) {
				float u = absU - centerU;
				float uu = u * u;
				float upperX = R_ * (u * A - v * B + mRsinBesinAl);
				float lowerX = sqrt(uu + vv + mR2);
				float upperY = R_ * (u * C - v * D - mRsinBecosAl);
				float lowerY = lowerX;
				float x = upperX / lowerX + centerX;
				float y = upperY / lowerY + centerY;
				int _v = centerV <= v ? v : v + centerV;
				int _u = centerU <= u ? u : u + centerU;
				mapX->at<float>(_v, _u) = x;
				mapY->at<float>(_v, _u) = y;
			}
		}



		/* .txt dosyasýndan alýnarak okunacaksa veri üstteki adýmlarýn hepsi koddan kaldýrýlabilir. sadece beta ve alpha deðerlerine göre koþul kontrolu yapýlcak ve .txt dosyasý okunacak. */

		printf("DEBUG*1");
		// Build map fonksiyonu için deðerlerimizi atýyoruz. 
		mapXs_[idx] = mapX;
		mapYs_[idx] = mapY;


		bitis = clock();
		printf("*****");
		cout << (float)(bitis - baslangic) / CLOCKS_PER_SEC;
	}

	void getImage(cv::Mat& src, cv::Mat& dest, int idx) {

		cv::remap(src, dest, *mapXs_[idx], *mapYs_[idx], cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
	}
};

int main(int ac, char* av[])
{
	cv::VideoCapture cap;
	cap.open("rtsp://192.168.1.10:554/user=admin&password=EXLXEXKX&channel=1&stream=.sdp");
	//cap.open(0);

	cv::Mat src_img;
	cv::Mat dest_img;

	cap >> src_img;

	cv::namedWindow("KAMERA");
	cv::namedWindow("ACROP_PANEL");

	FishEyeWindow few(1920,1080, 290, 850);
	float alpha = -270.0;
	float beta = 0.0;
	float theta = 270.0;
	float zoom = 0.55;


	few.buildMap(alpha, beta, theta, zoom);

	while (true) {

		cap >> src_img;
		clock_t baslangic = clock(), bitis;

		cv::resize(src_img, dest_img, cv::Size(), 1.5, 1.60, cv::INTER_CUBIC);

		few.getImage(dest_img, dest_img, 0);

		cv::putText(dest_img, //target image
			"ACROP", //text
			cv::Point(10, 20), //top-left position
			cv::FONT_HERSHEY_DUPLEX,
			0.6,
			CV_RGB(70, 255, 100), //font color
			1);



		cv::imshow("KAMERA", src_img);
		cv::Rect crop_region(0, 60, 290,790);
		cv::Mat cropped_image = dest_img(crop_region);

		cv::imshow("ACROP_PANEL", cropped_image);

		int key = cv::waitKey(1);



		bitis = clock();
		printf("---");
		cout << (float)(bitis - baslangic) / CLOCKS_PER_SEC;

		if (key == 27) {
			break;
		}
		switch (key) {
		case 'r':	zoom -= 0.01;
			cv::waitKey(7);
			few.buildMap(alpha, beta, theta, zoom);
			break;
		case 'f':	zoom += 0.01; 
			cv::waitKey(7);
			few.buildMap(alpha, beta, theta, zoom);
			break;
		case 'g':	alpha += 4;
			cv::waitKey(7);
			few.buildMap(alpha, beta, theta, zoom);
			break;
		case 't':	alpha -= 4;
			cv::waitKey(7);
			few.buildMap(alpha, beta, theta, zoom);
			break;
		case 'h':	beta += 4;
			cv::waitKey(7);
			few.buildMap(alpha, beta, theta, zoom);
			break;
		case 'y':	beta -= 4;
			cv::waitKey(7);
			few.buildMap(alpha, beta, theta, zoom);
			break;
		case 'j':	theta += 4; 
			cv::waitKey(7);
			few.buildMap(alpha, beta, theta, zoom);
			break;
		case 'u':	theta -= 4;
			cv::waitKey(7);
			few.buildMap(alpha, beta, theta, zoom);
			break;

		case 's':	cv::imwrite("./Acrop foto.png", dest_img); break;
		
		
	
		}
	}


	cap.release();
	cv::destroyAllWindows();

	return 0;
}