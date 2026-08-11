/*
 * scara_enable.exe — 直调 RobotSDK.dll，探测「使能 / 去使能 / 急停 / 清报警」
 *
 * 已确认（官方 UI 对照）:
 *   使能状态     = Read_PowerOn          (1=OFF, 2=ON)
 *   使能         = Send_PowerOn → Reset_PowerOn（脉冲，不可只按下不松开）
 *   去使能       = SetMode 来回切一次，最后回到原模式
 *   急停         = Send_Stop
 *   解除急停     = Reset_Stop
 *   清报警       = 先急停 OFF，再 Send_Reset→Reset_Reset（可两拍）
 *
 * 本 exe 不硬编码盘符：运行时用自身所在目录找 RobotSDK.dll / 工位配置。
 * 请把它放到与 RobotSDK.dll、许可证同目录（通常是 SNRobotLab 安装目录）。
 *
 * 用法:
 *   scara_enable.exe status|ping [--ip IP] [--port PORT]
 *   scara_enable.exe enable|disable [--ip IP] [--port PORT]
 *   scara_enable.exe set_mode <1|2|3|4> [--ip IP] [--port PORT]
 *   scara_enable.exe estop|release_estop|clear_alarm [--ip IP] [--port PORT]
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

typedef int (__cdecl *Fn_Read_PowerOn)(int *state, int connect_id);
typedef int (__cdecl *Fn_Send_PowerOn)(int connect_id);
typedef int (__cdecl *Fn_Reset_PowerOn)(int connect_id);
typedef int (__cdecl *Fn_ReadCurMode)(int *mode, int connect_id);
typedef int (__cdecl *Fn_SetMode)(int mode, int connect_id);
typedef int (__cdecl *Fn_Read_Stop)(int *state, int connect_id);
typedef int (__cdecl *Fn_Send_Stop)(int connect_id);
typedef int (__cdecl *Fn_Reset_Stop)(int connect_id);
typedef int (__cdecl *Fn_ClearAlarm)(int connect_id);
typedef int (__cdecl *Fn_Send_Reset)(int connect_id);
typedef int (__cdecl *Fn_Reset_Reset)(int connect_id);
typedef int (__cdecl *Fn_Read_Warning)(int *state, int connect_id);

typedef struct {
    HMODULE dll;
    Fn_RobotInit RobotInit;
    Fn_RobotRelease RobotRelease;
    Fn_ConnectTCP ConnectTCP;
    Fn_Disconnect Disconnect;
    Fn_Read_PowerOn Read_PowerOn;
    Fn_Send_PowerOn Send_PowerOn;
    Fn_Reset_PowerOn Reset_PowerOn;
    Fn_ReadCurMode ReadCurMode;
    Fn_SetMode SetMode;
    Fn_Read_Stop Read_Stop;
    Fn_Send_Stop Send_Stop;
    Fn_Reset_Stop Reset_Stop;
    Fn_ClearAlarm ClearAlarm;
    Fn_Send_Reset Send_Reset;
    Fn_Reset_Reset Reset_Reset;
    Fn_Read_Warning Read_Warning;
} Sdk;

static FARPROC must_get(HMODULE dll, const char *name) {
    FARPROC p = GetProcAddress(dll, name);
    if (!p) {
        fprintf(stderr, "ERR GetProcAddress(%s) failed\n", name);
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
        fprintf(stderr, "ERR missing %s (put scara_enable.exe in SNRobotLab)\n", dll_path);
        return 0;
    }
    SetDllDirectoryA(dir);
    sdk->dll = LoadLibraryA(dll_path);
    if (!sdk->dll) {
        fprintf(stderr, "ERR LoadLibrary GetLastError=%lu\n", GetLastError());
        return 0;
    }
    sdk->RobotInit = (Fn_RobotInit)(void *)must_get(sdk->dll, "RobotInit");
    sdk->RobotRelease = (Fn_RobotRelease)(void *)GetProcAddress(sdk->dll, "RobotRelease");
    sdk->ConnectTCP = (Fn_ConnectTCP)(void *)must_get(sdk->dll, "ConnectTCP");
    sdk->Disconnect = (Fn_Disconnect)(void *)must_get(sdk->dll, "Disconnect");
    sdk->Read_PowerOn = (Fn_Read_PowerOn)(void *)must_get(sdk->dll, "Read_PowerOn");
    sdk->Send_PowerOn = (Fn_Send_PowerOn)(void *)must_get(sdk->dll, "Send_PowerOn");
    sdk->Reset_PowerOn = (Fn_Reset_PowerOn)(void *)must_get(sdk->dll, "Reset_PowerOn");
    sdk->ReadCurMode = (Fn_ReadCurMode)(void *)must_get(sdk->dll, "ReadCurMode");
    sdk->SetMode = (Fn_SetMode)(void *)must_get(sdk->dll, "SetMode");
    sdk->Read_Stop = (Fn_Read_Stop)(void *)must_get(sdk->dll, "Read_Stop");
    sdk->Send_Stop = (Fn_Send_Stop)(void *)must_get(sdk->dll, "Send_Stop");
    sdk->Reset_Stop = (Fn_Reset_Stop)(void *)must_get(sdk->dll, "Reset_Stop");
    sdk->ClearAlarm = (Fn_ClearAlarm)(void *)must_get(sdk->dll, "ClearAlarm");
    sdk->Send_Reset = (Fn_Send_Reset)(void *)must_get(sdk->dll, "Send_Reset");
    sdk->Reset_Reset = (Fn_Reset_Reset)(void *)must_get(sdk->dll, "Reset_Reset");
    sdk->Read_Warning = (Fn_Read_Warning)(void *)must_get(sdk->dll, "Read_Warning");
    return 1;
}

static void sleep_ms(int ms) {
    if (ms <= 0) return;
    Sleep((DWORD)ms);
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

static const char *mode_name(int m) {
    switch (m) {
    case 1: return "T1";
    case 2: return "T2";
    case 3: return "Execute";
    case 4: return "Remote";
    default: return "?";
    }
}

static void print_status(Sdk *sdk, int cid) {
    int pwr = -1, stop = -1, mode = -1, warn = -1;
    int r1 = sdk->Read_PowerOn(&pwr, cid);
    int r2 = sdk->Read_Stop(&stop, cid);
    int r3 = sdk->Read_Warning(&warn, cid);
    int r4 = sdk->ReadCurMode(&mode, cid);
    printf("Read_PowerOn  rc=%d state=%d (%s)  << official UI enable\n",
           r1, pwr, pwr == 2 ? "ENABLE_ON" : (pwr == 1 ? "ENABLE_OFF" : "?"));
    printf("Read_Stop     rc=%d state=%d (%s)\n",
           r2, stop, stop == 2 ? "ESTOP_ON" : (stop == 1 ? "ESTOP_OFF" : "?"));
    printf("Read_Warning  rc=%d state=%d (%s)\n",
           r3, warn, warn == 2 ? "ALARM_ON" : (warn == 1 ? "ALARM_OFF" : "?"));
    printf("ReadCurMode   rc=%d mode=%d (%s)\n",
           r4, mode, mode_name(mode));
}

static int read_mode_pwr(Sdk *sdk, int cid, int *mode, int *pwr) {
    int m = -1, p = -1;
    sdk->ReadCurMode(&m, cid);
    sdk->Read_PowerOn(&p, cid);
    *mode = m;
    *pwr = p;
    return 0;
}

/* 使能前切到示教模式更稳；官方「执行」下也能使能，但实测 T1 最稳 */
static int ensure_teach_mode(Sdk *sdk, int cid) {
    int mode = -1, pwr = -1;
    read_mode_pwr(sdk, cid, &mode, &pwr);
    if (mode == 1 || mode == 2)
        return 0;
    int e = sdk->SetMode(1, cid);
    printf("ensure T1: SetMode(1)=%d (was mode %d)\n", e, mode);
    sleep_ms(400);
    return e;
}

static int wait_estop_off(Sdk *sdk, int cid, int timeout_ms) {
    int elapsed = 0;
    while (elapsed <= timeout_ms) {
        int stop = -1;
        sdk->Read_Stop(&stop, cid);
        if (stop == 1) {
            printf("wait_estop_off: ESTOP_OFF after %dms\n", elapsed);
            return 0;
        }
        sleep_ms(100);
        elapsed += 100;
    }
    printf("wait_estop_off: TIMEOUT still ESTOP_ON\n");
    return -1;
}

/* 报警复位脉冲：按下 Send_Reset → 松开 Reset_Reset */
static int pulse_alarm_reset(Sdk *sdk, int cid, const char *tag) {
    int e1 = sdk->Send_Reset(cid);
    printf("%s Send_Reset=%d\n", tag, e1);
    sleep_ms(200);
    int e2 = sdk->Reset_Reset(cid);
    printf("%s Reset_Reset=%d\n", tag, e2);
    sleep_ms(300);
    return (e1 != 0 && e2 != 0) ? -1 : 0;
}

static int clear_alarm_sequence(Sdk *sdk, int cid) {
    int stop = -1, warn = -1;
    sdk->Read_Stop(&stop, cid);
    sdk->Read_Warning(&warn, cid);

    if (stop == 2) {
        int e = sdk->Reset_Stop(cid);
        printf("Reset_Stop=%d\n", e);
        if (wait_estop_off(sdk, cid, 2000) != 0)
            return -1;
    } else {
        printf("ESTOP already OFF, skip Reset_Stop\n");
    }

    sdk->Read_Warning(&warn, cid);
    if (warn != 2) {
        printf("ALARM already OFF\n");
        return 0;
    }

    pulse_alarm_reset(sdk, cid, "pulse1");
    sdk->Read_Warning(&warn, cid);
    if (warn != 2) {
        printf("ALARM_OFF after pulse1\n");
        return 0;
    }

    pulse_alarm_reset(sdk, cid, "pulse2");
    sdk->Read_Warning(&warn, cid);
    if (warn != 2) {
        printf("ALARM_OFF after pulse2\n");
        return 0;
    }

    int e = sdk->ClearAlarm(cid);
    printf("ClearAlarm last-resort=%d\n", e);
    sleep_ms(200);
    pulse_alarm_reset(sdk, cid, "pulse3");
    sdk->Read_Warning(&warn, cid);
    return (warn == 2) ? -2 : 0;
}

/* 去使能：模式切出去再切回，最后停在原模式 */
static int disable_via_mode_transition(Sdk *sdk, int cid) {
    int mode = -1, pwr = -1;
    read_mode_pwr(sdk, cid, &mode, &pwr);
    printf("disable_path: mode=%d(%s) enable=%d\n", mode, mode_name(mode), pwr);
    if (pwr != 2) {
        printf("already ENABLE_OFF, nothing to do\n");
        return 0;
    }
    int orig = mode;
    int e = 0;
    if (mode == 1 || mode == 2) {
        e = sdk->SetMode(3, cid);
        printf("SetMode(3=Execute)=%d\n", e);
        sleep_ms(500);
        e = sdk->SetMode(orig, cid);
        printf("SetMode(%d=%s) back=%d\n", orig, mode_name(orig), e);
        sleep_ms(500);
    } else if (mode == 3) {
        e = sdk->SetMode(1, cid);
        printf("SetMode(1=T1)=%d\n", e);
        sleep_ms(500);
        e = sdk->SetMode(3, cid);
        printf("SetMode(3=Execute) back=%d\n", e);
        sleep_ms(500);
    } else {
        e = sdk->SetMode(1, cid);
        printf("SetMode(1=T1 fallback)=%d\n", e);
        sleep_ms(400);
        e = sdk->SetMode(orig, cid);
        printf("SetMode(%d) back=%d\n", orig, e);
        sleep_ms(500);
    }
    return e;
}

static void usage(void) {
    fprintf(stderr,
        "Usage:\n"
        "  scara_enable.exe status|ping [--ip IP] [--port PORT]\n"
        "  scara_enable.exe enable|disable [--ip IP] [--port PORT]\n"
        "  scara_enable.exe set_mode <1|2|3|4> [--ip IP] [--port PORT]\n"
        "  scara_enable.exe estop|release_estop|clear_alarm [--ip IP] [--port PORT]\n"
        "Close official GUI / snrobot serve first.\n");
}

int main(int argc, char **argv) {
    const char *cmd;
    int mode_arg = 0;
    char ip[64] = {0};
    int port = 0;
    int have_ip = 0;
    char dir[MAX_PATH] = {0};

    if (argc < 2) { usage(); return 1; }
    cmd = argv[1];

    int i = 2;
    if (strcmp(cmd, "set_mode") == 0) {
        if (argc < 3) { usage(); return 1; }
        mode_arg = atoi(argv[2]);
        if (mode_arg < 1 || mode_arg > 4) {
            fprintf(stderr, "ERR set_mode must be 1..4 (1=T1 2=T2 3=Execute 4=Remote)\n");
            return 1;
        }
        i = 3;
    } else if (strcmp(cmd, "status") && strcmp(cmd, "ping")
               && strcmp(cmd, "enable") && strcmp(cmd, "disable")
               && strcmp(cmd, "estop") && strcmp(cmd, "release_estop")
               && strcmp(cmd, "clear_alarm")) {
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
            fprintf(stderr, "ERR unknown arg: %s\n", argv[i]);
            return 1;
        }
    }

    if (!exe_dir(dir, sizeof(dir))) {
        fprintf(stderr, "ERR cannot get exe dir\n");
        return 2;
    }
    if (!SetCurrentDirectoryA(dir)) {
        fprintf(stderr, "ERR SetCurrentDirectory(%s) failed\n", dir);
        return 2;
    }

    if (!have_ip || port <= 0 || ip[0] == 0) {
        char cfg[MAX_PATH];
        path_join(cfg, sizeof(cfg), dir, "WorkStationConfig.json");
        if (!read_active_station(cfg, ip, sizeof(ip), &port)) {
            fprintf(stderr, "ERR cannot read %s; use --ip/--port\n", cfg);
            return 2;
        }
    }

    Sdk sdk = {0};
    if (!load_sdk(&sdk, dir)) return 2;

    int rc = sdk.RobotInit();
    if (rc != 0) {
        fprintf(stderr, "ERR RobotInit=%d\n", rc);
        return 3;
    }

    int cid = sdk.ConnectTCP(ip, (short)port);
    if (cid <= 0) {
        fprintf(stderr, "ERR ConnectTCP(%s,%d)=%d (official GUI / snrobot holding link?)\n",
                ip, port, cid);
        if (sdk.RobotRelease) sdk.RobotRelease();
        return 4;
    }
    printf("OK connect ip=%s port=%d id=%d\n", ip, port, cid);

    int fail = 0;
    if (strcmp(cmd, "ping") == 0) {
        printf("OK ping\n");
        print_status(&sdk, cid);
    } else if (strcmp(cmd, "status") == 0) {
        print_status(&sdk, cid);
        printf("OK status\n");
    } else if (strcmp(cmd, "enable") == 0) {
        print_status(&sdk, cid);
        ensure_teach_mode(&sdk, cid);
        int e1 = sdk.Send_PowerOn(cid);
        printf("Send_PowerOn=%d\n", e1);
        sleep_ms(200);
        int e2 = sdk.Reset_PowerOn(cid);
        printf("Reset_PowerOn (release request)=%d\n", e2);
        sleep_ms(500);
        print_status(&sdk, cid);
        if (e1 != 0) fail = 5; else printf("OK enable\n");
    } else if (strcmp(cmd, "disable") == 0) {
        print_status(&sdk, cid);
        int e = disable_via_mode_transition(&sdk, cid);
        print_status(&sdk, cid);
        if (e != 0) fail = 5; else printf("OK disable\n");
    } else if (strcmp(cmd, "set_mode") == 0) {
        print_status(&sdk, cid);
        int e = sdk.SetMode(mode_arg, cid);
        printf("SetMode(%d=%s)=%d\n", mode_arg, mode_name(mode_arg), e);
        print_status(&sdk, cid);
        if (e != 0) fail = 5; else printf("OK set_mode\n");
    } else if (strcmp(cmd, "estop") == 0) {
        print_status(&sdk, cid);
        int e = sdk.Send_Stop(cid);
        printf("Send_Stop=%d\n", e);
        print_status(&sdk, cid);
        if (e != 0) fail = 5; else printf("OK estop\n");
    } else if (strcmp(cmd, "release_estop") == 0) {
        print_status(&sdk, cid);
        int e = sdk.Reset_Stop(cid);
        printf("Reset_Stop=%d\n", e);
        sleep_ms(300);
        print_status(&sdk, cid);
        if (e != 0) fail = 5; else printf("OK release_estop\n");
    } else if (strcmp(cmd, "clear_alarm") == 0) {
        print_status(&sdk, cid);
        int e = clear_alarm_sequence(&sdk, cid);
        print_status(&sdk, cid);
        if (e == -1) {
            fail = 5;
            printf("FAIL clear_alarm: estop still ON\n");
        } else if (e == -2) {
            fail = 5;
            printf("FAIL clear_alarm: ALARM still ON\n");
        } else {
            printf("OK clear_alarm\n");
        }
    }

    sdk.Disconnect(cid);
    if (sdk.RobotRelease) sdk.RobotRelease();
    return fail;
}
