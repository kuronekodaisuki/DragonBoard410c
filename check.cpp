#include <stdio.h>
#include <stdexcept>
#include <GL/gl.h>
#include <GL/glx.h>

#include "check.h"

// Check saticefy OvrvisionPro
bool OvrvisionProOpenCL::CheckGPU()
{
	bool result = false;
	cl_uint num_of_platforms = 0;
	// get total number of available platforms:
	cl_int err = clGetPlatformIDs(0, 0, &num_of_platforms);
	SAMPLE_CHECK_ERRORS(err);

	vector<cl_platform_id> platforms(num_of_platforms);
	// get IDs for all platforms:
	err = clGetPlatformIDs(num_of_platforms, &platforms[0], 0);
	SAMPLE_CHECK_ERRORS(err);

	for (cl_uint i = 0; i < num_of_platforms; i++)
	{
		char devicename[80];
		clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(devicename), devicename, NULL);
		printf("PLATFORM: %s\n", devicename);

		clGetGLContextInfoKHR_fn pclGetGLContextInfoKHR = NULL;
		pclGetGLContextInfoKHR = GETFUNCTION(platforms[i], clGetGLContextInfoKHR);

		cl_context_properties opengl_props[] = {
			CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[i],
			CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(),
			CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(),
			0
		};

		size_t devSizeInBytes = 0;
		pclGetGLContextInfoKHR(opengl_props, CL_DEVICES_FOR_GL_CONTEXT_KHR, 0, NULL, &devSizeInBytes);
		const size_t devNum = devSizeInBytes / sizeof(cl_device_id);
		if (devNum)
		{
			std::vector<cl_device_id> devices(devNum);
			pclGetGLContextInfoKHR(opengl_props, CL_DEVICES_FOR_GL_CONTEXT_KHR, devSizeInBytes, &devices[0], NULL);
			for (size_t k = 0; k < devNum; k++)
			{
				cl_device_type t;
				clGetDeviceInfo(devices[k], CL_DEVICE_TYPE, sizeof(t), &t, NULL);
				//if (t == CL_DEVICE_TYPE_GPU)
				{
					clGetDeviceInfo(devices[k], CL_DEVICE_NAME, sizeof(devicename), devicename, NULL);
					char buffer[32];
					clGetDeviceInfo(devices[k], CL_DEVICE_OPENCL_C_VERSION, sizeof(buffer), buffer, NULL);
					printf("\t%s %s\n", devicename, buffer);
				}
			}
		}

		// Check Memory capacity and extensions
		bool gl_sharing = false, version = false, d3d11_sharing = false, memory = false;
		cl_ulong mem_size;
		cl_uint num_of_devices = 0;
		if (CL_SUCCESS == clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, 0, &num_of_devices))
		{
			cl_device_id *id = new cl_device_id[num_of_devices];
			err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, num_of_devices, id, 0);
			SAMPLE_CHECK_ERRORS(err);
			for (cl_uint j = 0; j < num_of_devices; j++)
			{
				clGetDeviceInfo(id[j], CL_DEVICE_NAME, sizeof(devicename), devicename, NULL);
				printf("\tGPU: %s\n", devicename);

				// Check version
				char buffer[32];
				if (clGetDeviceInfo(id[j], CL_DEVICE_OPENCL_C_VERSION, sizeof(buffer), buffer, NULL) == CL_SUCCESS)
				{
					if (strcmp(buffer, "OpenCL C 1.2") >= 0)
					{
						version = true;
					}
					printf("\tOpenCL: %s\n", buffer);
				}

				// Check memory capacity
				clGetDeviceInfo(id[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
				printf("\tGLOBAL_MEM_SIZE: %ld MBytes\n", mem_size / (1024 * 1024));
				clGetDeviceInfo(id[j], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(mem_size), &mem_size, NULL);
				printf("\tMAX_MEM_ALLOC_SIZE: %ld MBytes\n", mem_size / (1024 * 1024));
				// TODO: Determine which is dominant
				if (256 <= mem_size / (1024 * 1024))
				{
					memory = true;
				}

				// Check extensions
				size_t size;
				char *extensions;
				clGetDeviceInfo(id[j], CL_DEVICE_EXTENSIONS, 0, NULL, &size); // get entension size
				extensions = new char[size];

				clGetDeviceInfo(id[j], CL_DEVICE_EXTENSIONS, size, extensions, NULL);
				if (strstr(extensions, "cl_khr_gl_sharing") != NULL)
				{
					gl_sharing = true;
					printf("\tcl_khr_gl_sharing\n");
				}

#ifdef _DEBUG
				puts(extensions);
#endif		

			}

			if (gl_sharing && memory && version)
			{
				result = true;
				printf("\tOvrvisionPro: Positive\n\n");
			}

			else if (256 <= mem_size / (1024 * 1024))
			{
				printf("\tOvrvisionPro: Depend on resolution\n\n");
			}
			else
			{
				printf("\tOvrvisionPro: Negative\n\n");
			}
		}
	}
	return result;
}