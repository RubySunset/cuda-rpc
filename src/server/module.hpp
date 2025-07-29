#include <sys/stat.h>
#include <stdlib.h>
#include <queue>
#include <chrono>
#include <fractos/service/compute/cuda.hpp>


namespace impl {
    class Context;
    class Function;
}

namespace impl {

    class Module {
    public:
        CUmodule get_remote_cumodule() const;

        static std::shared_ptr<Module> factory(std::string& name, CUcontext& ctx);
        static std::shared_ptr<Module> factory(CUcontext& ctx, std::shared_ptr<char[]>& buffer, size_t size
                                                   , std::weak_ptr<Context> vctx);

        fractos::core::future<void> register_methods(std::shared_ptr<fractos::core::channel> ch);

    protected:
        void handle_generic(auto ch, auto args);
        void handle_get_global(auto ch, auto args);
        void handle_get_function(auto ch, auto args);

        void handle_destroy(auto args);
    
    private:
        void module_unload();  

        // TODO: weak_ptr
        std::shared_ptr<Module> _self;
        // TODO: atomic_flag
        bool _destroyed;
        // TODO: delete
        std::string _name;
        uint64_t _id;
    public:
        CUcontext _ctx;
        CUmodule _module;
    private:
        std::weak_ptr<Context> _vctx;
        std::shared_ptr<const char[]> _data;

    public:
        fractos::core::cap::request _req_generic;
        fractos::core::cap::request _req_destroy;

        std::shared_ptr<Function> _func; 

        Module(std::string& name, CUcontext& ctx);
        Module(CUcontext& ctx,  std::shared_ptr<char[]>& buffer, size_t size, std::weak_ptr<Context> vctx);

        ~Module();

        //std::vector<std::shared_ptr<gpu_device_memory>> allocations;
    };

    std::string to_string(const Module& obj);

}
