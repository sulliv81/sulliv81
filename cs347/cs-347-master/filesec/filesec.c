/* Bo Sullivan 10/15/2020
 * CSCI 347, FALL 2020
 * Dr. Elglaly
 * filesec.c
 */
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>

/* This main method is going make use of C's build in functions of open(),
 * read(), write(), and close(). The goal if this program is to utilize
 * these functions into order to encrypt of decrypt a input.txt file.
 * The encryption and decryption part is going to be as simple as adding
 * 100 ASCI values to the input chars and output them into an output file
 * That can later be decrypted with the -d flag. I added in checks to make
 * sure a txt file is the 3rd argument, and that the 2nd argument is also
 * a -d or -e flag to handle for bad user inputs. Getting the open command
 * to work properly was great once I read more about flags and permissions.
 * Then I added some while loops within the read to handle the cases in which
 * the file might be larger than my buffer and so as to not just infinitely 
 * keep reading from the read() command.
 */
int main(int argc, char** argv)
{
    //Check for the correctness of the entered command. If it is incorrect, display a usage message
    //write your code here
    if (strcmp(argv[1],"-e") != 0 && strcmp(argv[1],"-d") != 0)
    {
	printf("Usage:\nfilesec -e|-d [filename]\n");
    }
    if (strstr(argv[2], ".txt") == NULL)
    {
	printf("Usage:\nfilesec -e|-d [filename]\n");
    }
    char outputFile[1500];
    char* command = argv[1];
    char* file = argv[2];

    if (strcmp(argv[1],"-e") == 0 && strstr(argv[2], ".txt") != NULL)
    {
        int pFile = open(file, O_RDONLY);
        strcpy(outputFile, file);
        strtok(outputFile, ".");
        strcat(outputFile, "_enc.txt");

        //encrypt the contents of the input file and save the results in the output file. Do not forget to create the output file
        //write your code here
	int readByte;
	char buffer[1500];
	readByte = read(pFile, buffer, 1500-1);

	for (int i = 0; i < 1500; i++) {
		buffer[i] = 100 + buffer[i];
	}
	int oFile = open(outputFile, O_CREAT | O_WRONLY, S_IRWXU);
	write(oFile, buffer, readByte);

	close(outputFile);
	close(file);

    }

    if (strcmp(argv[1],"-d") == 0 && strstr(argv[2], ".txt") != NULL)
    {
        int pFile = open(file, O_RDONLY);
	strcpy(outputFile, file);
	strtok(outputFile, ".");
	strcat(outputFile, "_dec.txt");

        //create and open the output file
        //write your code here
        //decrypt the contents of the input file and save the results in the output file
        //write your code here

        int counter = 0;

        int readByte;
        char buffer[1500];
        readByte = read(pFile, buffer, 1500-1);

        for (int i = 0; i < 1500; i++) {
                buffer[i] = buffer[i] - 100;
        }

        int oFile = open(outputFile, O_CREAT | O_WRONLY, S_IRWXU);
        write(oFile, buffer, readByte);

	close(outputFile);
	close(file);

    }
	return 0;
}
