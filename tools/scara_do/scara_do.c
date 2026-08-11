/*
 * scara_do.exe — 用户输出 DO（方案 C）
 *
 * 从 RobotSDK.dll 调用的函数：
 *   连接：RobotInit, ConnectTCP, Disconnect, RobotRelease（可选）
 *   写 DO：WriteUOEnable(ch, 0) 保持用户模式 U
 *         WriteUO(ch, 0|1)       写 UO 电平（0 或 1，由命令行传入）
 * 以后可在本 exe 里继续加其它 SDK 导出函数（读 IO、其它 UO 等）。
 *
 * 本 exe 不硬编码盘符：运行时用自身所在目录找 RobotSDK.dll / 工位配置。
 * 请把它放到与 RobotSDK.dll、许可证同目录（通常是 SNRobotLab 安装目录）。
 * 用法: scara_do.exe set <通道> <0|1> [--ip IP] [--port PORT]
 *       scara_do.exe ping [--ip IP] [--port PORT]
 */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int (__cdecl *Fn_RobotInit)(void);
typedef int (__cdecl *Fn_RobotRelease)(void);
typedef int (__cdecl *Fn_ConnectTCP)(const char *ip, short port);
typedef int (__cdecl *Fn_Disconnect)(int connect_id);
typedef int (__cdecl *Fn_WriteUOEnable)(int io_map, int enable, int connect_id);
typedef int (__cdecl *Fn_WriteUO)(int io_map, unsigned char io_status, int connect_id);

typedef struct {
    HMODULE dll;
    Fn_RobotInit RobotInit;
    Fn_RobotRelease RobotRelease;
    Fn_ConnectTCP ConnectTCP;
    Fn_Disconnect Disconnect;
    Fn_WriteUOEnable WriteUOEnable;
    Fn_WriteUO WriteUO;
} Sdk;

static FARPROC must_get(HMODULE dll, const char *name) {
    FARPROC p = GetProcAddress(dll, name);
    if (!p) {
        fprintf(stderr, "ERR GetProcAddress(%s) 失败\n", name);
        exit(2);
    }
    return p;
}

static void path_join(char *out, size_t n, const char *dir, const char *file) {
    size_t dlen = strlen(dir);
    if (dlen && (dir[dlen - 1] == '\\' || dir[dlen - 1] == '/'))
        snprintf(out, n, "%s%s", dir, file);
    else
        snprintf(out, n, "%s\\%s", dir, file);
}

/* RobotSDK 以宿主 exe 目录为配置根 → sdk_dir = 本 exe 所在目录 */
static int exe_dir(char *out, size_t n) {
    char path[MAX_PATH] = {0};
    if (!GetModuleFileNameA(NULL, path, MAX_PATH))
        return 0;
    char *slash = strrchr(path, '\\');
    if (!slash)
        slash = strrchr(path, '/');
    if (!slash)
        return 0;
    *slash = 0;
    strncpy(out, path, n - 1);
    out[n - 1] = 0;
    return 1;
}

static int load_sdk(Sdk *sdk, const char *dir) {
    char dll_path[MAX_PATH];
    path_join(dll_path, sizeof(dll_path), dir, "RobotSDK.dll");
    if (GetFileAttributesA(dll_path) == INVALID_FILE_ATTRIBUTES) {
        fprintf(stderr, "ERR 找不到 %s（请把 scara_do.exe 放到 SNRobotLab）\n", dll_path);
        return 0;
    }
    SetDllDirectoryA(dir);
    sdk->dll = LoadLibraryA(dll_path);
    if (!sdk->dll) {
        fprintf(stderr, "ERR LoadLibrary 失败 GetLastError=%lu\n", GetLastError());
        return 0;
    }
    sdk->RobotInit = (Fn_RobotInit)(void *)must_get(sdk->dll, "RobotInit");
    sdk->RobotRelease = (Fn_RobotRelease)(void *)GetProcAddress(sdk->dll, "RobotRelease");
    sdk->ConnectTCP = (Fn_ConnectTCP)(void *)must_get(sdk->dll, "ConnectTCP");
    sdk->Disconnect = (Fn_Disconnect)(void *)must_get(sdk->dll, "Disconnect");
    sdk->WriteUOEnable = (Fn_WriteUOEnable)(void *)must_get(sdk->dll, "WriteUOEnable");
    sdk->WriteUO = (Fn_WriteUO)(void *)must_get(sdk->dll, "WriteUO");
    return 1;
}

static int read_active_station(const char *cfg_path, char *ip, size_t ip_n, int *port) {
    FILE *f = fopen(cfg_path, "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz > 1 << 20) { fclose(f); return 0; }
    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return 0; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { free(buf); fclose(f); return 0; }
    buf[sz] = 0;
    fclose(f);

    const char *active = strstr(buf, "\"Active\"");
    while (active) {
        const char *colon = strchr(active, ':');
        if (colon && strstr(colon, "true") == colon + 1 + strspn(colon + 1, " \t\r\n"))
            break;
        active = strstr(active + 1, "\"Active\"");
    }
    if (!active) { free(buf); return 0; }

    const char *obj = active;
    while (obj > buf && *obj != '{') obj--;

    const char *ip_key = NULL, *port_key = NULL;
    for (const char *p = obj; p < active; p++) {
        if (strncmp(p, "\"IP\"", 4) == 0) ip_key = p;
        if (strncmp(p, "\"Port\"", 6) == 0) port_key = p;
    }
    if (!ip_key || !port_key) { free(buf); return 0; }

    const char *q1 = strchr(ip_key, '"');
    q1 = strchr(q1 + 1, '"');
    q1 = strchr(q1 + 1, '"');
    const char *q2 = strchr(q1 + 1, '"');
    if (!q1 || !q2 || (size_t)(q2 - q1 - 1) >= ip_n) { free(buf); return 0; }
    memcpy(ip, q1 + 1, (size_t)(q2 - q1 - 1));
    ip[q2 - q1 - 1] = 0;

    const char *pv = strchr(port_key, ':');
    if (!pv) { free(buf); return 0; }
    while (*pv && (*pv < '0' || *pv > '9') && *pv != '"') pv++;
    if (*pv == '"') pv++;
    *port = atoi(pv);
    free(buf);
    return (*port > 0 && ip[0] != 0);
}

static void usage(void) {
    fprintf(stderr,
        "用法:\n"
        "  scara_do.exe set <通道> <0|1> [--ip IP] [--port PORT]\n"
        "  scara_do.exe ping [--ip IP] [--port PORT]\n");
}

int main(int argc, char **argv) {
    const char *cmd;
    int ch = 1, on = 1;
    char ip[64] = {0};
    int port = 0;
    int have_ip = 0;
    char dir[MAX_PATH] = {0};

    if (argc < 2) { usage(); return 1; }
    cmd = argv[1];

    int i = 2;
    if (strcmp(cmd, "set") == 0) {
        if (argc < 4) { usage(); return 1; }
        ch = atoi(argv[2]);
        on = atoi(argv[3]) ? 1 : 0;
        i = 4;
    } else if (strcmp(cmd, "ping") == 0) {
        i = 2;
    } else {
        usage();
        return 1;
    }

    for (; i < argc; i++) {
        if (strcmp(argv[i], "--ip") == 0 && i + 1 < argc) {
            strncpy(ip, argv[++i], sizeof(ip) - 1);
            have_ip = 1;
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = atoi(argv[++i]);
            have_ip = 1;
        } else {
            fprintf(stderr, "ERR 未知参数: %s\n", argv[i]);
            return 1;
        }
    }

    if (!exe_dir(dir, sizeof(dir))) {
        fprintf(stderr, "ERR 无法取得 exe 目录\n");
        return 2;
    }
    if (!SetCurrentDirectoryA(dir)) {
        fprintf(stderr, "ERR SetCurrentDirectory(%s) 失败\n", dir);
        return 2;
    }

    if (!have_ip || port <= 0 || ip[0] == 0) {
        char cfg[MAX_PATH];
        path_join(cfg, sizeof(cfg), dir, "WorkStationConfig.json");
        if (!read_active_station(cfg, ip, sizeof(ip), &port)) {
            fprintf(stderr, "ERR 无法读 %s，请用 --ip/--port\n", cfg);
            return 2;
        }
    }

    Sdk sdk = {0};
    if (!load_sdk(&sdk, dir)) return 2;

    int rc = sdk.RobotInit();
    if (rc != 0) {
        fprintf(stderr, "ERR RobotInit=%d（exe 须在 SNRobotLab，且许可有效）\n", rc);
        return 3;
    }

    int cid = sdk.ConnectTCP(ip, (short)port);
    if (cid <= 0) {
        fprintf(stderr, "ERR ConnectTCP(%s,%d)=%d（官方软件/snrobot 是否占用连接？）\n",
                ip, port, cid);
        if (sdk.RobotRelease) sdk.RobotRelease();
        return 4;
    }
    printf("OK connect ip=%s port=%d id=%d\n", ip, port, cid);

    if (strcmp(cmd, "ping") == 0) {
        printf("OK ping\n");
    } else {
        int e1 = sdk.WriteUOEnable(ch, 0, cid);
        int e2 = sdk.WriteUO(ch, (unsigned char)on, cid);
        printf("WriteUOEnable(UO,%d,0)=%d\n", ch, e1);
        printf("WriteUO(UO,%d,%d)=%d\n", ch, on, e2);
        if (e1 != 0 || e2 != 0) {
            sdk.Disconnect(cid);
            if (sdk.RobotRelease) sdk.RobotRelease();
            return 5;
        }
        printf("OK set UO[%d]=%d\n", ch, on);
    }

    sdk.Disconnect(cid);
    if (sdk.RobotRelease) sdk.RobotRelease();
    return 0;
}
