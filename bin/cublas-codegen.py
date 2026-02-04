import os
import re
import subprocess
import shutil
import argparse
from dataclasses import dataclass, field

import clang.cindex
from clang.cindex import CursorKind, TypeKind


@dataclass
class Arg:
    name: str
    type: str
    is_pointer: bool
    pointee_type: str | None
    is_void_ptr: bool = field(init=False)
    type_nonconst: str = field(init=False)
    pointee_type_nonconst: str | None = field(init=False)

    def __post_init__(self):
        if self.is_pointer:
            assert self.pointee_type is not None
        self.is_void_ptr = self.type in ("void *", "const void *")
        self.type_nonconst = self.type.replace("const", "")
        self.pointee_type_nonconst = self.pointee_type.replace("const", "") if self.is_pointer else None


class CodeBuilder:

    """
    API functions which we choose to generate.
    If the whitelist is empty, all valid functions will be generated.
    """
    WHITELIST = {
        # Unit test-specific functions
        "cublasSaxpy_v2",
        "cublasSgemv_v2",
        "cublasSgemm_v2",
        "cublasSgeam",

        # llama.cpp-specific functions
        "cublasSetMathMode",
        "cublasGemmStridedBatchedEx",
        "cublasGemmBatchedEx",
        "cublasGemmEx",
    }

    """
    Some CUBLAS parameters of type const void* or void* have information about
    their size contained in another enum argument.
    This is a mapping from names of these parameters to the name of the enum
    parameter that contains this size information.
    These variable-size parameters may be pointers to host or device memory.
    """
    SIZE_MAPPING = {
        "alpha": "computeType",
        "beta": "computeType",
    }

    def __init__(self, cublas_header_path: str, manual_symbol_path: str, wrapper_dir: str, server_dir: str) -> None:
        print(f"Runnning CUBLAS code builder using {cublas_header_path}, skipping functions defined in {manual_symbol_path}, writing to wrapper dir {wrapper_dir} and server dir {server_dir}.")

        self.wrapper_code = []
        self.server_code = []
        self.handler_id_map = {} # function handler name -> id
        self.func_freq = {} # function frequency (for overloads)
        self.func_counter = 0 # counter for unique function ids

        whitelist = self.WHITELIST
        blacklist = self.extract_blacklist(manual_symbol_path)

        self.gen_init()

        index = clang.cindex.Index.create()
        tu = index.parse(cublas_header_path, args=['-x', 'c++'])

        for cursor in tu.cursor.walk_preorder():

            if cursor.kind != CursorKind.FUNCTION_DECL:
                continue

            func_name = cursor.spelling

            if (
                (whitelist and func_name not in whitelist) or # if there is a whitelist, skip non-whitelisted functions
                not self.is_cublas_func(func_name) or # skip non-cublas API functions
                func_name in blacklist or # skip blacklisted functions
                cursor.is_definition() # skip defined functions
            ):
                continue

            # Handle function overloads
            if func_name in self.handler_id_map:
                self.func_freq[func_name] += 1
                handler_name = func_name + "_overload" + str(self.func_freq[func_name])
                self.handler_id_map[handler_name] = self.func_counter
            else:
                self.func_freq[func_name] = 1
                handler_name = func_name
                self.handler_id_map[func_name] = self.func_counter

            print(func_name)

            self.gen_func_start(func_name, handler_name, cursor)
            self.rpc_client_args = []
            self.cublas_server_args = []

            for i, clang_arg in enumerate(cursor.get_arguments()):

                if i == 0: # skip cublas handle argument (in every cublas API function)
                    continue

                arg = Arg(
                    name=clang_arg.spelling,
                    type=self.convert_array_to_pointer(clang_arg.type.spelling),
                    is_pointer=(clang_arg.type.kind == TypeKind.POINTER),
                    pointee_type=(clang_arg.type.get_pointee().spelling if clang_arg.type.kind == TypeKind.POINTER else None),
                )

                # If the type is const void* or void* and the name matches an entry
                # in SIZE_MAPPING, we need to recover it's size from the associated
                # enum
                if arg.is_void_ptr and arg.name in self.SIZE_MAPPING:
                    self.gen_with_size_mapping(arg)

                # If the type is const void* or void*, but doesn't match an entry
                # in SIZE_MAPPING, it must be a pointer to device memory
                # (e.g. a tensor), so we can directly copy it
                # Alternatively, if it's not a pointer type at all, then we can
                # also just copy it. The logic is exactly the same in either case.
                elif arg.is_void_ptr or not arg.is_pointer:
                    self.gen_direct_copy(arg)

                # If pointer type to non-void, can be to host or device memory
                # - handle both cases
                else:
                    self.gen_ambig_ptr(arg)

            self.gen_func_end(func_name)

        self.gen_server_router()

        with open(f"{wrapper_dir}/cublas-codegen.cpp", "w+") as f:
            f.write("\n".join(self.wrapper_code))
        with open(f"{server_dir}/cublas-codegen.cpp", "w+") as f:
            f.write("\n".join(self.server_code))

        if shutil.which("clang-format") is None:
            print("Warning: clang-format not found, skipping formatting")
            return

        subprocess.run(["clang-format", "-i", "-style={BasedOnStyle: llvm, IndentWidth: 4}", "src/wrapper/cublas-autogen.cpp"])
        subprocess.run(["clang-format", "-i", "-style={BasedOnStyle: llvm, IndentWidth: 4}", "src/server/cublas-autogen.hpp"])

    def is_cublas_func(self, func_name: str) -> bool:
        return len(func_name) >= 6 and func_name[:6] == "cublas"

    def convert_array_to_pointer(self, type: str) -> str:
        """
        Convert the string representation of an array type
        (from libclang) to an equivalent pointer type.
        """
        return re.sub(r'\[\]', r'*', type)

    def extract_blacklist(self, file_path: str) -> set[str]:
        """
        Using the provided file of manually defined CUBLAS symbols,
        generate a blacklist of functions we cannot auto-generate.
        """
        index = clang.cindex.Index.create()
        tu = index.parse(file_path, args=['-x', 'c++'])
        blacklist = set()
        for cursor in tu.cursor.walk_preorder():
            if (cursor.kind == CursorKind.FUNCTION_DECL and
                os.path.abspath(cursor.location.file.name) == os.path.abspath(file_path) # ignore functions from includes
            ):
                blacklist.add(cursor.spelling)
        return blacklist

    def gen_init(self) -> None:
        self.wrapper_code.append("""
            // Autogenerated by cublas-autogen.py.

            #include "fractos/service/compute/cuda.hpp"
            #include <cuda_runtime.h>
            #include <cublas_v2.h>
            #include <fractos/logging.hpp>

            #include "cublas-state.hpp"
            #include "magic_enum.hpp"

            #include <functional>
            #include <optional>

            namespace clt = fractos::service::compute::cuda;

        """)
        self.server_code.append("""
            // Autogenerated by cublas-autogen.py.

            #include <cuda_runtime.h>
            #include <cublas_v2.h>
            #include <fractos/logging.hpp>

        """)

    def gen_func_start(self, func_name: str, handler_name: str, cursor: "CursorKind") -> None:
        self.wrapper_code.append(f"""
            extern "C" [[gnu::visibility("default")]]
            cublasStatus_t
            {func_name}({", ".join([f"{self.convert_array_to_pointer(arg.type.spelling)} {arg.spelling}" for arg in cursor.get_arguments()])})
            {{
                auto& cublas_state = get_cublas_state();

                auto cublas_ptr = cublas_state.get_handle(handle);
                if (cublas_ptr == nullptr) [[unlikely]] {{
                    return CUBLAS_STATUS_NOT_INITIALIZED;
                }}

                [[maybe_unused]] auto& driver_state = get_driver_state_return_cublas();
        """)
        self.server_code.append(f"""
            cublasStatus_t
            handle_{handler_name}(const char* args_ptr, cublasHandle_t handle)
            {{
        """)

    def gen_with_size_mapping(self, arg: Arg) -> None:
        self.wrapper_code.append(f"""
            bool {arg.name}_is_host;
            uint8_t {arg.name}_size = 0;
            uint64_t {arg.name}_arg;
            if (driver_state.get_memory(reinterpret_cast<uintptr_t>({arg.name})) == nullptr) {{
                {arg.name}_is_host = true;
                auto enum_name = magic_enum::enum_name({self.SIZE_MAPPING[arg.name]});
                std::string buf = "";
                for (char c : enum_name) {{
                    if (c >= '0' && c <= '9') {{
                        buf.push_back(c);
                    }} else if (!buf.empty()) {{
                        break;
                    }}
                }}
                {arg.name}_size = std::stoi(buf);
                {arg.name}_size >>= 3;
                std::memcpy(&{arg.name}_arg, {arg.name}, {arg.name}_size);
            }} else {{
                {arg.name}_is_host = false;
                std::memcpy(&{arg.name}_arg, &{arg.name}, sizeof(void*));
            }}
        """)
        self.rpc_client_args += [f"{arg.name}_is_host", f"{arg.name}_size", f"{arg.name}_arg"]

        self.server_code.append(f"""
            bool {arg.name}_is_host = *reinterpret_cast<const bool*>(args_ptr);
            args_ptr += sizeof(bool);
            uint8_t {arg.name}_size = *reinterpret_cast<const uint8_t*>(args_ptr);
            args_ptr += sizeof(uint8_t);
            void* {arg.name}_ptr;
            [[maybe_unused]] uint64_t {arg.name}_data;
            if ({arg.name}_is_host) {{
                std::memcpy(&{arg.name}_data, args_ptr, {arg.name}_size);
                {arg.name}_ptr = &{arg.name}_data;
            }} else {{
                std::memcpy(&{arg.name}_ptr, args_ptr, sizeof(void*));
            }}
            args_ptr += sizeof(uint64_t);
        """)
        self.cublas_server_args.append(f"{arg.name}_ptr")

    def gen_direct_copy(self, arg: Arg) -> None:
        self.rpc_client_args.append(arg.name)
        self.server_code.append(f"""
            {arg.type_nonconst} {arg.name};
            std::memcpy(&{arg.name}, args_ptr, sizeof({arg.type}));
            args_ptr += sizeof({arg.type});
        """)
        self.cublas_server_args.append(arg.name)

    def gen_ambig_ptr(self, arg: Arg) -> None:
        self.wrapper_code.append(f"""
            bool {arg.name}_is_host;
            uint64_t {arg.name}_arg;
            if (driver_state.get_memory(reinterpret_cast<uintptr_t>({arg.name})) == nullptr) {{
                {arg.name}_is_host = true;
                std::memcpy(&{arg.name}_arg, {arg.name}, sizeof(*{arg.name}));
            }} else {{
                {arg.name}_is_host = false;
                std::memcpy(&{arg.name}_arg, &{arg.name}, sizeof({arg.name}));
            }}
        """)
        self.rpc_client_args += [f"{arg.name}_is_host", f"{arg.name}_arg"]

        self.server_code.append(f"""
            bool {arg.name}_is_host = *reinterpret_cast<const bool*>(args_ptr);
            args_ptr += sizeof(bool);
            {arg.type_nonconst} {arg.name}_ptr;
            [[maybe_unused]] {arg.pointee_type_nonconst} {arg.name}_data;
            if ({arg.name}_is_host) {{
                std::memcpy(&{arg.name}_data, args_ptr, sizeof({arg.pointee_type_nonconst}));
                {arg.name}_ptr = &{arg.name}_data;
            }} else {{
                std::memcpy(&{arg.name}_ptr, args_ptr, sizeof({arg.type_nonconst}));
            }}
            args_ptr += sizeof(uint64_t);
        """)
        self.cublas_server_args.append(f"{arg.name}_ptr")

    def gen_func_end(self, func_name: str) -> None:
        self.wrapper_code.append(f"""
            auto stream_ptr = cublas_state.get_stream(handle);
            std::optional<std::reference_wrapper<clt::Stream>> stream_wrapper =
                (stream_ptr == nullptr) ?
                std::optional<std::reference_wrapper<clt::Stream>>{{}} :
                *stream_ptr;
            try {{
                cublas_ptr->autogen_func({self.func_counter}, stream_wrapper, {", ".join(self.rpc_client_args)}).get();
                return CUBLAS_STATUS_SUCCESS;
            }} catch (const srv::CublasError& e) {{
                return e.cublas_error;
            }}
        }}

        """)
        self.server_code.append(f"""
            return {func_name}(handle, {", ".join(self.cublas_server_args)});
        }}

        """)

        self.func_counter += 1

    def gen_server_router(self) -> None:
        self.server_code.append(f"""
            bool route_autogen_func(uint32_t func_id, const char* args_ptr, cublasHandle_t handle, cublasStatus_t* cublas_error) {{
                switch (func_id) {{
                    {"\n".join([f"case {handler_id}: *cublas_error = handle_{handler_name}(args_ptr, handle); return true;" for handler_name, handler_id in self.handler_id_map.items()])}
                    default: return false;
                }}
            }}
        """)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cublas-header")
    parser.add_argument("--cublas-manual-file")
    parser.add_argument("--wrapper-dir")
    parser.add_argument("--server-dir")
    args = parser.parse_args()
    builder = CodeBuilder(os.path.abspath(args.cublas_header), os.path.abspath(args.cublas_manual_file), os.path.abspath(args.wrapper_dir), os.path.abspath(args.server_dir))
