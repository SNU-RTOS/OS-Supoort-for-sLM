#!/usr/bin/env python3
import os
import mmap
import time
import subprocess

FILE_PATH = "test_16GB.bin"
FILE_SIZE = 32 * 1024 * 1024 * 1024  # 16GB
BLOCK_SIZE = 4096

def drop_caches():
    """
    Linux에서 페이지 캐시를 비우는 함수.
    관리자 권한(또는 sudo) 필요.
    """
    try:
        # sync로 디스크 반영 후 drop_caches=3
        subprocess.run(["sudo", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[WARNING] drop_caches 실패: {e}")
        print("sudo 권한이 없거나, Linux 환경이 아닐 수 있습니다.")

def create_4gb_file(path, size=FILE_SIZE):
    """
    4GB 짜리 파일 생성 (간단히 truncate).
    """
    print(f"Creating file {path} ({size} bytes)...")
    with open(path, "wb") as f:
        f.truncate(size)
    print("File creation complete.\n")

def get_current_memory_usage_mb():
    """
    현재 프로세스가 사용하는 물리 메모리(RSS)를 MB 단위로 반환.
    /proc/self/status에서 VmRSS 항목을 파싱 (Linux 전용).
    """
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # 예: VmRSS:    123456 kB
                    parts = line.split()
                    if len(parts) >= 3:
                        rss_kb = float(parts[1])  # 단위: kB
                        return rss_kb / 1024.0    # MB로 변환
    except FileNotFoundError:
        # 혹은 다른 OS 등
        pass
    return 0.0

def measure_mmap_io(path):
    """
    mmap 기반으로 파일을 전부 읽는 데 걸리는 시간을 측정.
    """
    start = time.time()
    
    with open(path, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        
        total_read = 0
        while total_read < FILE_SIZE:
            _ = mm[total_read:total_read + BLOCK_SIZE]
            total_read += BLOCK_SIZE
            
        end = time.time()
        print(f"Current RSS usage: {get_current_memory_usage_mb():.2f} MB\n")
        mm.close()
    
    return end - start

def measure_read_io(path):
    """
    일반적인 read() 기반 I/O로 파일을 전부 읽는 데 걸리는 시간 측정.
    """
    start = time.time()
    
    with open(path, "rb") as f:
        while True:
            data = f.read(BLOCK_SIZE)
            if not data:
                break
            
    end = time.time()
    
    print(f"Current RSS usage: {get_current_memory_usage_mb():.2f} MB\n")
    return end - start

def measure_direct_io(path):
    """
    Direct I/O (O_DIRECT) 방식으로 파일을 전부 읽는 데 걸리는 시간을 측정.
    버퍼 정렬 및 크기 주의 필요.
    """
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    
    # Direct I/O는 버퍼가 블록 크기에 정렬되어 있어야 함
    buf = bytearray(BLOCK_SIZE)
    buf_view = memoryview(buf)
    
    start = time.time()
    total_read = 0
    while True:
        # os.readv()의 반환값은 '읽은 총 바이트 수(int)'
        bytes_read = os.readv(fd, [buf_view])
        if bytes_read == 0:
            break
        total_read += bytes_read
    
    end = time.time()
    
    print(f"Current RSS usage: {get_current_memory_usage_mb():.2f} MB\n")
    
    os.close(fd)
    return end - start

def run_io_test(test_name, func, drop_cache_first=True):
    """
    주어진 I/O 테스트 함수(func)를 호출하고,
    걸린 시간을 출력하는 헬퍼 함수.
    """
    print(f"\n=== {test_name} ===")
    
    if drop_cache_first:
        drop_caches()
        time.sleep(5)  # 캐시가 완전히 비워지도록 잠깐 대기
    
    elapsed = func(FILE_PATH)
    print(f"{test_name} time: {elapsed:.2f} sec")
    
def main():
    # 1. 4GB 파일 존재 검사, 없으면 생성
    if not os.path.exists(FILE_PATH):
        create_4gb_file(FILE_PATH, FILE_SIZE)
    else:
        print("4GB 테스트 파일이 이미 존재합니다.\n")
    
    drop_caches()  # 캐시 비우기
    print(f"Current RSS usage: {get_current_memory_usage_mb():.2f} MB\n")
    # 2. 테스트 시나리오 실행
    #    - (with cache) 테스트는 drop_caches=False 로 호출
    run_io_test("mmap I/O test (cold)", measure_mmap_io, drop_cache_first=True)
    run_io_test("mmap I/O test (with cache)", measure_mmap_io, drop_cache_first=False)
    run_io_test("mmap I/O test (with cache)", measure_mmap_io, drop_cache_first=False)
    
    
    run_io_test("read() I/O test (cold)", measure_read_io, drop_cache_first=True)
    run_io_test("read() I/O test (with cache)", measure_read_io, drop_cache_first=False)
    run_io_test("read() I/O test (with cache)", measure_read_io, drop_cache_first=False)
    
    
    run_io_test("Direct I/O test (cold)", measure_direct_io, drop_cache_first=True)
    run_io_test("Direct I/O test (with cache)", measure_direct_io, drop_cache_first=False)

if __name__ == "__main__":
    main()
