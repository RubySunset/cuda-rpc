#ifdef cuGetProcAddress
#undef cuGetProcAddress
#endif

#ifdef cuDeviceGetUuid
#undef cuDeviceGetUuid
#endif

#ifdef cuDeviceTotalMem
#undef cuDeviceTotalMem
#endif

#ifdef cuMemFree
#undef cuMemFree
#endif

SYM(cuGetErrorName)
SYM(cuGetErrorString)
SYM(cuGetExportTable)
SYM(cuGetProcAddress_v2)
