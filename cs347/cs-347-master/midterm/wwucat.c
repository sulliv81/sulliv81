/* Bo Sullivan
 * CSCI 347 Dr. Elglaly
 * Midterm Fall 2020
 * wwucat.c
*/
#include <fcntl.h>
#include <unistd.h>              
#include <string.h>
#include <stdio.h>

#define BUF_SIZE 32               /* Size of buffer */

#define FILE_NAME_LEN 200         /* Maximum length of file name */

/* This main method will take a file as input and concatenate the
 * contents of the file to additional argument txt files that are
 * passed. If no arguments passed, error messages will print. Program
 * makes use of open, read, write, and close which error message 
 * printing if any issues are encountered there. For loop to handle
 * any length of argc passed.
*/
int main(int argc, char *argv[])
{
	if (argc < 2){
		 printf("Expecting 1 filename\n");
	}

	char buffer[BUF_SIZE+1];

	for (int i = 1; i < argc; i++) {

		int pFile = open(argv[i], O_RDONLY);

		if(pFile == -1){
			printf("Error while opening the file for reading.");
		}
		size_t readByte = read(pFile, buffer, FILE_NAME_LEN);

		if(readByte == -1){
			 printf("Error while reading the file.");
		}
		buffer[readByte] = '\0';

		int writeByte = write(fileno(stdout), buffer, readByte);

		if (writeByte == -1){
			 printf("Error while writing to standard output.");
		}
		close(argv[i]);
	}
    return 0;
}

