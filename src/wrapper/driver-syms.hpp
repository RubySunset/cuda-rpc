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

#ifdef cuMemcpyDtoD_v2
#undef cuMemcpyDtoD_v2
#endif

#ifdef cuMemcpyDtoH_v2
#undef cuMemcpyDtoH_v2
#endif

#ifdef cuMemcpyHtoD_v2
#undef cuMemcpyHtoD_v2
#endif

#ifdef cuModuleGetGlobal_v2
#undef cuModuleGetGlobal_v2
#endif

SYM(cuGetErrorName)
SYM(cuGetErrorString)
SYM(cuGetExportTable)
SYM(cuGetProcAddress_v2)
