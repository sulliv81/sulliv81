/* Bo Sullivan CSCI 347
 * Fall 2020, Dr. Elglaly
 * Filestat.c 10/21/2020
 */

#include <unistd.h>        /* F_OK, STDIN_FILENO, STDERR_FILENO, etc. */
#include <sys/stat.h>      /* struct stat */
#include <sys/types.h>     /* S_IFMT */
#include <stdio.h>         /* fprintf, printf, sprintf, etc. */
#include <stdlib.h>        /* exit, etc. */
#include <time.h>          /* ctime */

/* This main method will run a file or directory input from the command line and then
 * it will declare the file type, the date of last access, and the permissions for user,
 * group, and others. This will be acheived through the use of c's stat() feature.
 * A quick check to make sure the input is correct and if so, then resume the task of
 * printing the aforementioned goals while using stat and inode system calls.
 */
int main(int argc, char *argv[])

{
    mode_t file_perm;
    struct stat file_details;  /* Detailed file info */
    char* file = argv[1];

    /* Retrieve the file details */
   //write your code here

    stat(file, &file_details);
    if (!S_ISDIR(file_details.st_mode) && !S_ISREG(file_details.st_mode)) {

	printf("Usage:\nfilestat <field|dir>");
    }
    else {

    /* Get the file type */
    //write your code here
    printf("File type: %s", (S_ISDIR(file_details.st_mode)) ? "Directory\n" : "Regular");
    /* Get the time of last access of the file */
    //write your code here
    printf("\nTime of last access: %s", ctime(&file_details.st_atime));
    /* Get the file permissions */
   // stat(file, &file_perm)
    printf("File Permissions:\n");
    //write your code here
    printf("User: %s", (file_details.st_mode & S_IRUSR) ? "Readable" : "Not readable"); 
    printf(", %s", (file_details.st_mode & S_IWUSR) ? "Writable" : "Not writable");
    printf(", %s", (file_details.st_mode & S_IXUSR) ? "Executable" : "Not executable");
    printf("\nGroup: %s", (file_details.st_mode & S_IRGRP) ? "Readable" : "Not Readable");
    printf(", %s", (file_details.st_mode & S_IWGRP) ? "Writable" : "Not writable");
    printf(", %s", (file_details.st_mode & S_IXGRP) ? "Executable" : "Not executable");
    printf("\nOthers: %s", (file_details.st_mode & S_IROTH) ? "Readable" : "Not Readable");
    printf(", %s", (file_details.st_mode & S_IWOTH) ? "Writable" : "Not writable");
    printf(", %s", (file_details.st_mode & S_IXOTH) ? "Executable" : "Not executable\n");

    }
}
