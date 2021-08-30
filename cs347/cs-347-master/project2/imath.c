#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <pthread.h>
#include <string.h>
#define THREADS 4

#define filterWidth 3
#define filterHeight 3

#define RGB_MAX 255

pthread_barrier_t barrier;
//pthread_mutex_t mymutex;
typedef struct {
	 unsigned char r, g, b;
} PPMPixel;

struct parameter {
	PPMPixel *image;         //original image
	PPMPixel *result;        //filtered image
	unsigned long int w;     //width of image
	unsigned long int h;     //height of image
	unsigned long int start; //starting point of work
	unsigned long int size;  //equal share of work (almost equal if odd)
};


/*This is the thread function. It will compute the new values for the region of image specified in params (start to start+size) using convolution.
    (1) For each pixel in the input image, the filter is conceptually placed on top ofthe image with its origin lying on that pixel.
    (2) The  values  of  each  input  image  pixel  under  the  mask  are  multiplied  by the corresponding filter values.
    (3) The results are summed together to yield a single output value that is placed in the output image at the location of the pixel being processed on the input.
 
 */


void *threadfn(void *params)
{
//	pthread_mutex_lock(&mymutex);
	int laplacian[filterWidth][filterHeight] =
	{
	  -1, -1, -1,
	  -1,  8, -1,
	  -1, -1, -1,
	};
	struct parameter *myp;// = (struct parameter*)params;
	myp = (struct parameter*)params;

    /*For all pixels in the work region of image (from start to start+size)
      Multiply every value of the filter with corresponding image pixel. Note: this is NOT matrix multiplication.
      Store the new values of r,g,b in p->result.
     */
	for (int x = 0; x < myp->w; x++) {
		for (int y = myp->start; y < myp->start+myp->size; y++) {
			int red = 0;
			int green = 0;
			int blue = 0;

			for(int filterx = 0; filterx < filterWidth; filterx++) {
				for(int filtery = 0; filtery < filterHeight; filtery++) {

					int imagex = (x - (filterWidth/2) + filterx + myp->w) % myp->w;
					int imagey = (y - (filterHeight/2) + filtery + myp->h) % myp->h;
					red += myp->image[imagey*myp->w+imagex].r * laplacian[filterx][filtery];
                                        green += myp->image[imagey*myp->w+imagex].g * laplacian[filterx][filtery];
                                        blue += myp->image[imagey*myp->w+imagex].b * laplacian[filterx][filtery];
				}
			}
			if (red < 0){ red = 0;}
                        if (red > 255) {red = 255;}

			if (blue < 0) {blue = 0;}
                        if (blue > 255){ blue = 255;}

			if (green > 255){ green = 255;}
                        if (green < 0) {green = 0;}


			myp->result[y * myp->w + x].r = red;
			myp->result[y * myp->w + x].g = green;
			myp->result[y * myp->w + x].b = blue;
		}
	}
//	pthread_mutex_unlock(&mymutex);
//	pthread_exit(NULL);
	return NULL;
}



/*Create a new P6 file to save the filtered image in. Write the header block
 e.g. P6
      Width Height
      Max color value
 then write the image data.
 The name of the new file shall be "name" (the second argument).
*/
void writeImage(PPMPixel *imgg, char *name, unsigned long int width, unsigned long int height)
{
	FILE *wFile;
	char newName[128];
	strcpy(newName, name);
	strtok(newName, ".");
	strcat(newName, "_laplacian.ppm");
	wFile = fopen(newName, "wb");

	if (!wFile) {
		printf("could not open to write binary");
		exit(1);
	}
	fprintf(wFile, "P6\n");
//	width = imgg->w;
//	height = imgg->h;
	fprintf(wFile,"%ld %ld\n", width, height);

	fprintf(wFile,"%d\n", 255);

//	fprintf(wFile, imgg->image);

    	fwrite(imgg, 1, 3*width*height, wFile);
	
	pclose(wFile);
}

/* Open the filename image for reading, and parse it.
    Example of a ppm header:    //http://netpbm.sourceforge.net/doc/ppm.html
    P6                  -- image format
    # comment           -- comment lines begin with
    ## another comment  -- any number of comment lines
    200 300             -- image width & height
    255                 -- max color value
 */

	


 /*
 Check if the image format is P6. If not, print invalid format error message.
 Read the image size information and store them in width and height.
 Check the rgb component, if not 255, display error message.
 Return: pointer to PPMPixel that has the pixel data of the input image (filename)
 */
PPMPixel *readImage(const char *filename, unsigned long int *width, unsigned long int *height)
{

	struct parameter *img;
//	img = img->image;
	//read image format starting with buffer the size of 1 byte

	char  buffer[32];
	FILE *pFile = fopen(filename, "rb"); // or "rb" for read binary and or fopen
	
	if (!pFile) {
		printf("error opening with fOpen");
	}
	char  c;// = {'\0'};
	
	char rgb[10];
	int r = 0;

	//p6
	fscanf(pFile, "%s", buffer);
	if (!fgets(buffer, sizeof(buffer), pFile)) {
		printf("error\n");
		exit(1);
	}
	//img = malloc(sizeof(struct parameter));



	//readByte = fread(buffer, sizeof(char), 2, pFile);


/*
	if (strcmp(buffer, "P6") != 0) {
		printf("Not P6\n");
		printf(buffer);
		printf("\n");
//		printf("%s\n%s\n", buffer[0], buffer[1]);
		exit(1);
	}
	if (strcmp(buffer, "P6") == 0) {
		printf("Is P6\n");
	}
*/
	img = malloc(sizeof(struct parameter));

	if (!img) {
		printf("Bad malloc\n");
		exit(1);
	}


	c = fgetc(pFile);
	while (c == '#') {
		while(c != '\n') {
			c = fgetc(pFile);
		}
	}
	if (fscanf(pFile, "%ld %ld", &img->w, &img->h) != 2) {
		printf("not enough args\n");
		exit(1);
	}
      else {
		printf("Width: %ld\n", img->w);
		printf("Height: %ld\n", img->h);
	}
        *width = img->w;
        *height = img->h;

	if (fscanf(pFile, "%s", rgb) != 1) {
		printf("invalid args or invalid count");
	} else {
		printf("%d\n", RGB_MAX);
	}
	if (atoi(rgb) != 255) {
		printf("not 255");
		exit(1);
	} 


	fgetc(pFile);
	//fscanf(pFile, "%ld %ld %d", &img->w, &img->h, &r);
	//if (r != 255) { printf(" no\n");}
	img->image = malloc(img->w * img->h * sizeof(PPMPixel));

	if (!img) {
		printf("memory troubles");
	}
	fread(img->image,  1, img->w*img->h*3, pFile) ;
	fclose(pFile);
	//check the image format by reading the first two characters in filename and compare them to P6.


	//If there are comments in the file, skip them. You may assume that comments exist only in the header block.

	
	//read image size information
	

	//Read rgb component. Check if it is equal to RGB_MAX. If  not, display error message.
	
    
    //allocate memory for img. NOTE: A ppm image of w=200 and h=300 will contain 60000 triplets (i.e. for r,g,b), ---> 18000 bytes.

    //read pixel data from filename into img. The pixel data is stored in scanline order from left to right (up to bottom) in 3-byte chunks (r g b values for each pixel) encoded as binary numbers.
//	for (int i = 478601; i < 478602; i++) {
//		printf("red: %d green: %d blue: %d\n", img->image[i].r, img->image[i].g, img->image[i].b);
//	}


	return img->image;
}

/* Create threads and apply filter to image.
 Each thread shall do an equal share of the work, i.e. work=height/number of threads.
 Compute the elapsed time and store it in *elapsedTime (Read about gettimeofday).
 Return: result (filtered image)
 */
PPMPixel *apply_filters(PPMPixel *image, unsigned long w, unsigned long h, double *elapsedTime) {

//    pthread_mutex_init(&mymutex, NULL);
   	PPMPixel *result;
	result = malloc(w * h * 3);
	pthread_barrier_init(&barrier, NULL, THREADS);


	struct timeval start, end;
	
// struct parameter *result = malloc(sizeof *result);

	gettimeofday(&start, NULL);


//	*elapsedTime = end_time.tv_sec - start_time.tv_sec;
  

  //allocate memory for result
    //allocate memory for parameters (one for each thread)

    int totalWork = h / THREADS;
    pthread_t threads[THREADS];
    //struct parameter array[THREADS];

     struct parameter *array = malloc(sizeof(struct parameter) * THREADS);

//	struct parameter *array;

//    struct parameter *array[THREADS];
	
	//1array[0] = malloc(sizeof(struct param
    for (int i = 0; i < THREADS; i++) {
//	array[i] = *(struct parameter*)malloc(sizeof(struct parameter));
	array[i].image = image;
	array[i].result = result;
	array[i].w = w;
	array[i].h = h;
 	array[i].start = totalWork * i;
        if (i == 3) {array[i].size = h - (totalWork*i); }
	else {array[i].size = totalWork; }
	pthread_create(&threads[i], NULL, threadfn,&array[i]);
	printf("thread created\n");
    }
       printf("3\n");

    for (int j = 0; j < THREADS; j++) {
        void* val;
	pthread_join(threads[j], &val);
    }

    /*create threads and apply filter.
     For each thread, compute where to start its work.  Determine the size of the work. If the size is not even, the last thread shall take the rest of the work.
     
	*/

   //Let threads wait till they all finish their work.

//	pthread_mutex_destroy(&mymutex);
//	pthread_exit(NULL);

	gettimeofday(&end, NULL);
	*elapsedTime += (end.tv_usec + end.tv_usec) - (start.tv_sec * start.tv_usec)  / 1000000.0;
	return result;

}


/*The driver of the program. Check for the correct number of arguments. If wrong print the message: "Usage ./a.out filename"
    Read the image that is passed as an argument at runtime. Apply the filter. Print elapsed time in .3 precision (e.g. 0.006 s). Save the result image in a file called laplacian.ppm. Free allocated memory.
 */
int main(int argc, char *argv[])
{
	//load the image into the buffer
    unsigned long int w, h;
    double elapsedTime = 0.0;
    struct parameter *img;

    if (argc < 2 || argc > 3) {
	printf("Usage ./a.out filename.ppm\n");
    }

    if (argc == 2) {
	img->image = readImage(argv[1], &w, &h);
	//img = readImage(argv[1], &w, &h);
//        for (int j = 0; j < 10; j++) {
//                printf("red: %d green: %d green: %d\n", img->image[j].r, img->image[j].g, img->image[j].b);
//        }

	img->result = apply_filters(img->image, w, h, &elapsedTime);
	//result = apply_filters(img, w, h, &elapsedTime);
//	for (int i = 0; i < 10; i++) {
//		printf("red: %d green: %d blue: %d\n", img->result[i].r, img->result[i].g, img->result[i].b);
//	}
	writeImage(img->result, argv[1], w, h); 
	//writeImage(result, argv[1], w, h);

	printf("%f\n", elapsedTime);
    }

	return 0;
}















