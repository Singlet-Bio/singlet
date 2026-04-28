// test_sra_fifo.cpp — Test native SRA reader through FIFO architecture
//
// This tests the exact same code path as singlify's SRA mode:
//   SRA file → VDB API → SraReader → FILE* fdopen → Named FIFOs → Reader captures
//
// Usage: ./test_sra_fifo FILE.sra
// Writes R1.fastq and R2.fastq to /tmp/sra_fifo_test/

#include "sra_reader.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s FILE.sra\n", argv[0]);
        return 1;
    }

    const char* sra_path = argv[1];
    const char* test_dir = "/tmp/sra_fifo_test";
    mkdir(test_dir, 0755);

    // Create FIFOs (matching singlify architecture)
    char r1_fifo[256], r2_fifo[256];
    snprintf(r1_fifo, sizeof(r1_fifo), "%s/R1.fifo", test_dir);
    snprintf(r2_fifo, sizeof(r2_fifo), "%s/R2.fifo", test_dir);
    unlink(r1_fifo);
    unlink(r2_fifo);

    if (mkfifo(r1_fifo, 0600) != 0 || mkfifo(r2_fifo, 0600) != 0) {
        fprintf(stderr, "mkfifo failed: %s\n", strerror(errno));
        return 1;
    }

    // Readiness pipe (same as singlify)
    int ready_pipe[2];
    pipe(ready_pipe);

    pid_t writer_pid = fork();
    if (writer_pid == 0) {
        // ── WRITER CHILD: SRA → FIFOs (same pattern as singlify demuxer) ──
        close(ready_pipe[0]);

        int r1_fd = open(r1_fifo, O_RDWR);  // O_RDWR trick
        int r2_fd = open(r2_fifo, O_RDWR);
        if (r1_fd < 0 || r2_fd < 0) {
            char fail = 0;
            write(ready_pipe[1], &fail, 1);
            _exit(1);
        }

#ifdef __linux__
        fcntl(r1_fd, F_SETPIPE_SZ, 1 << 20);
        fcntl(r2_fd, F_SETPIPE_SZ, 1 << 20);
#endif

        char ok = 1;
        write(ready_pipe[1], &ok, 1);
        close(ready_pipe[1]);

        FILE* r1_fp = fdopen(r1_fd, "w");
        FILE* r2_fp = fdopen(r2_fd, "w");
        static char r1_buf[1 << 20], r2_buf[1 << 20];
        setvbuf(r1_fp, r1_buf, _IOFBF, sizeof(r1_buf));
        setvbuf(r2_fp, r2_buf, _IOFBF, sizeof(r2_buf));

        SraReader reader;
        reader.open(sra_path);
        fprintf(stderr, "[writer] %lld spots to write\n", (long long)reader.total_spots());

        while (reader.write_next_spot(r1_fp, r2_fp)) {}

        fflush(r1_fp); fclose(r1_fp);
        fflush(r2_fp); fclose(r2_fp);
        fprintf(stderr, "[writer] Done: %lld spots\n", (long long)reader.spots_read());
        _exit(0);
    }

    // ── PARENT: wait for FIFOs to be ready, then fork readers ──
    close(ready_pipe[1]);
    char ready;
    if (read(ready_pipe[0], &ready, 1) != 1 || ready != 1) {
        fprintf(stderr, "Writer failed to open FIFOs\n");
        return 1;
    }
    close(ready_pipe[0]);

    // Fork reader for R1 FIFO
    pid_t r1_reader = fork();
    if (r1_reader == 0) {
        char out_path[256];
        snprintf(out_path, sizeof(out_path), "%s/R1.fastq", test_dir);
        int fifo_fd = open(r1_fifo, O_RDONLY);
        int out_fd = open(out_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        char buf[1 << 20];
        ssize_t n;
        while ((n = read(fifo_fd, buf, sizeof(buf))) > 0)
            write(out_fd, buf, n);
        close(fifo_fd);
        close(out_fd);
        _exit(0);
    }

    // Fork reader for R2 FIFO
    pid_t r2_reader = fork();
    if (r2_reader == 0) {
        char out_path[256];
        snprintf(out_path, sizeof(out_path), "%s/R2.fastq", test_dir);
        int fifo_fd = open(r2_fifo, O_RDONLY);
        int out_fd = open(out_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        char buf[1 << 20];
        ssize_t n;
        while ((n = read(fifo_fd, buf, sizeof(buf))) > 0)
            write(out_fd, buf, n);
        close(fifo_fd);
        close(out_fd);
        _exit(0);
    }

    // Wait for all children
    int status;
    waitpid(writer_pid, &status, 0);
    int writer_exit = WIFEXITED(status) ? WEXITSTATUS(status) : 1;

    waitpid(r1_reader, &status, 0);
    waitpid(r2_reader, &status, 0);

    // Cleanup FIFOs
    unlink(r1_fifo);
    unlink(r2_fifo);

    if (writer_exit != 0) {
        fprintf(stderr, "Writer failed with exit code %d\n", writer_exit);
        return 1;
    }

    fprintf(stderr, "\n[test] FIFO pipe test complete.\n");
    fprintf(stderr, "[test] R1: %s/R1.fastq\n", test_dir);
    fprintf(stderr, "[test] R2: %s/R2.fastq\n", test_dir);
    fprintf(stderr, "[test] Compare with:\n");
    fprintf(stderr, "  awk 'NR%%4==2' %s/R1.fastq | md5sum\n", test_dir);
    fprintf(stderr, "  awk 'NR%%4==2' <fastq-dump-R1>  | md5sum\n");
    return 0;
}
