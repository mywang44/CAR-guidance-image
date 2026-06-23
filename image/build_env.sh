conda create -n car_guidance_image python=3.10

conda activate car_guidance_image

conda install numpy=1.26.4 pytorch=2.1.1 torchvision=0.16.1 pytorch-cuda=11.8 -c pytorch -c nvidia

conda install ninja=1.12.1 absl-py=2.1.0

conda install lpips=0.1.3 ml-collections=0.1.1 openai-clip=1.0.1

# If you hit:
#   ImportError: .../libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent
# Some environments/channels don't ship the runtime libittnotify.so. This creates
# a minimal stub and preloads it on conda activate.
mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d" "${CONDA_PREFIX}/etc/conda/deactivate.d"
gcc -shared -fPIC -Wl,-soname,libittnotify.so -x c -o "${CONDA_PREFIX}/lib/libittnotify.so" - <<'EOF'
#ifdef __cplusplus
extern "C" {
#endif
int iJIT_NotifyEvent(int event_type, void *event_specific_data) {
    (void)event_type;
    (void)event_specific_data;
    return 0;
}
int iJIT_IsProfilingActive(void) { return 0; }
unsigned int iJIT_GetNewMethodID(void) { static unsigned int id = 1; return id++; }
#ifdef __cplusplus
}
#endif
EOF
cat > "${CONDA_PREFIX}/etc/conda/activate.d/ittnotify_preload.sh" <<'EOF'
#!/usr/bin/env bash
export _FLOWGRAD_OLD_LD_PRELOAD="${LD_PRELOAD-}"
export LD_PRELOAD="${CONDA_PREFIX}/lib/libittnotify.so${LD_PRELOAD:+:${LD_PRELOAD}}"
EOF
cat > "${CONDA_PREFIX}/etc/conda/deactivate.d/ittnotify_preload.sh" <<'EOF'
#!/usr/bin/env bash
export LD_PRELOAD="${_FLOWGRAD_OLD_LD_PRELOAD-}"
unset _FLOWGRAD_OLD_LD_PRELOAD
EOF

conda install numpy=1.26.4 pytorch=2.4.1 torchvision=0.19.1 pytorch-cuda=12.4 lpips=0.1.3 openai-clip=1.0.1 ml-collections=0.1.1 absl-py=2.1.0 ninja=1.12.1 -c pytorch -c nvidia
