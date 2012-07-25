#!/bin/bash

gcc -DCLS=$(getconf LEVEL1_DCACHE_LINESIZE) sse-test.c -o sse-test.exe
