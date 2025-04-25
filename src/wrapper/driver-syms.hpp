#ifdef cuGetProcAddress
#undef cuGetProcAddress
#endif

#ifdef cuDeviceGetUuid
#undef cuDeviceGetUuid
#endif

#ifdef cuDeviceTotalMem
#undef cuDeviceTotalMem
#endif

SYM(cuGetErrorName)
SYM(cuGetErrorString)
SYM(cuGetExportTable)
SYM(cuGetProcAddress_v2)
SYM(cuInit)
