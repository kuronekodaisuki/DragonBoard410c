//
//
//
#include <stdio.h>
#include <sys/stat.h>
#include <opencv2/core/core.hpp>

#include <CL/cl.h>	// OpenCL and its extensions
#include <CL/cl_ext.h>
#include <CL/cl_gl.h>       // OpenCL/OpenGL interoperabillity
#include <CL/cl_gl_ext.h>   // OpenCL/OpenGL interoperabillity

typedef unsigned int TEXTURE;

using namespace std;

string opencl_error_to_str(cl_int error)
{
#define CASE_CL_CONSTANT(NAME) case NAME: return #NAME;

	// Suppose that no combinations are possible.
	// TODO: Test whether all error codes are listed here
	switch (error)
	{
		CASE_CL_CONSTANT(CL_SUCCESS)
		CASE_CL_CONSTANT(CL_DEVICE_NOT_FOUND)
		CASE_CL_CONSTANT(CL_DEVICE_NOT_AVAILABLE)
		CASE_CL_CONSTANT(CL_COMPILER_NOT_AVAILABLE)
		CASE_CL_CONSTANT(CL_MEM_OBJECT_ALLOCATION_FAILURE)
		CASE_CL_CONSTANT(CL_OUT_OF_RESOURCES)
		CASE_CL_CONSTANT(CL_OUT_OF_HOST_MEMORY)
		CASE_CL_CONSTANT(CL_PROFILING_INFO_NOT_AVAILABLE)
		CASE_CL_CONSTANT(CL_MEM_COPY_OVERLAP)
		CASE_CL_CONSTANT(CL_IMAGE_FORMAT_MISMATCH)
		CASE_CL_CONSTANT(CL_IMAGE_FORMAT_NOT_SUPPORTED)
		CASE_CL_CONSTANT(CL_BUILD_PROGRAM_FAILURE)
		CASE_CL_CONSTANT(CL_MAP_FAILURE)
		CASE_CL_CONSTANT(CL_MISALIGNED_SUB_BUFFER_OFFSET)
		CASE_CL_CONSTANT(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST)
		CASE_CL_CONSTANT(CL_INVALID_VALUE)
		CASE_CL_CONSTANT(CL_INVALID_DEVICE_TYPE)
		CASE_CL_CONSTANT(CL_INVALID_PLATFORM)
		CASE_CL_CONSTANT(CL_INVALID_DEVICE)
		CASE_CL_CONSTANT(CL_INVALID_CONTEXT)
		CASE_CL_CONSTANT(CL_INVALID_QUEUE_PROPERTIES)
		CASE_CL_CONSTANT(CL_INVALID_COMMAND_QUEUE)
		CASE_CL_CONSTANT(CL_INVALID_HOST_PTR)
		CASE_CL_CONSTANT(CL_INVALID_MEM_OBJECT)
		CASE_CL_CONSTANT(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR)
		CASE_CL_CONSTANT(CL_INVALID_IMAGE_SIZE)
		CASE_CL_CONSTANT(CL_INVALID_SAMPLER)
		CASE_CL_CONSTANT(CL_INVALID_BINARY)
		CASE_CL_CONSTANT(CL_INVALID_BUILD_OPTIONS)
		CASE_CL_CONSTANT(CL_INVALID_PROGRAM)
		CASE_CL_CONSTANT(CL_INVALID_PROGRAM_EXECUTABLE)
		CASE_CL_CONSTANT(CL_INVALID_KERNEL_NAME)
		CASE_CL_CONSTANT(CL_INVALID_KERNEL_DEFINITION)
		CASE_CL_CONSTANT(CL_INVALID_KERNEL)
		CASE_CL_CONSTANT(CL_INVALID_ARG_INDEX)
		CASE_CL_CONSTANT(CL_INVALID_ARG_VALUE)
		CASE_CL_CONSTANT(CL_INVALID_ARG_SIZE)
		CASE_CL_CONSTANT(CL_INVALID_KERNEL_ARGS)
		CASE_CL_CONSTANT(CL_INVALID_WORK_DIMENSION)
		CASE_CL_CONSTANT(CL_INVALID_WORK_GROUP_SIZE)
		CASE_CL_CONSTANT(CL_INVALID_WORK_ITEM_SIZE)
		CASE_CL_CONSTANT(CL_INVALID_GLOBAL_OFFSET)
		CASE_CL_CONSTANT(CL_INVALID_EVENT_WAIT_LIST)
		CASE_CL_CONSTANT(CL_INVALID_EVENT)
		CASE_CL_CONSTANT(CL_INVALID_OPERATION)
		CASE_CL_CONSTANT(CL_INVALID_GL_OBJECT)
		CASE_CL_CONSTANT(CL_INVALID_BUFFER_SIZE)
		CASE_CL_CONSTANT(CL_INVALID_MIP_LEVEL)
		CASE_CL_CONSTANT(CL_INVALID_GLOBAL_WORK_SIZE)
		CASE_CL_CONSTANT(CL_INVALID_PROPERTY)
	default:
		return "UNKNOWN ERROR CODE ";// +to_str(error);
	}

#undef CASE_CL_CONSTANT
}

#define SAMPLE_CHECK_ERRORS(ERR)  \
	if (ERR != CL_SUCCESS) { \
		printf("%s %d\n", __FILE__, __LINE__); \
		throw std::runtime_error(opencl_error_to_str(ERR).c_str()); \
		}

#define GETFUNCTION(platform, x) \
    (x ## _fn)clGetExtensionFunctionAddressForPlatform(platform, #x);

// OpenCL Sharing mode 
enum SHARING_MODE {
	NONE = 0,
	OPENGL,
	D3D11,
	//D3D9
};

// Scaling
enum SCALING {
	HALF,	// 1/2
	FOURTH,	// 1/4
	EIGHTH	// 1/8
};

// Extension vendor 
enum VENDOR {
	KHRONOS,	// Khronos specific extension
	INTELGPU,	// Intel specific extension
	AMD,		// AMD specific extension
	NVIDIA		// NVIDIA specific extension
};

// Filter mode
enum FILTER {
	RAW,
	GAUSSIAN,
	MEDIAN,
	GAUSSIAN_3,
	GAUSSIAN_5,
	GAUSSIAN_7,
	MEDIAN_3,
	MEDIAN_5,
	MEDIAN_7,
};

// Convex object
//typedef struct {
//	int mx, my;				// mass center
//   std::vector<cv::Point> convexs;	// convex contor
//} Convex;

// OpenCL extension callback function
typedef int(*EXTENSION_CALLBACK)(void *pItem, const char *extensions);

// OpenCL version
class OvrvisionProOpenCL {
public:
	/*! @brief Constructor
	@param width of image
	@param height of image
	@param mode of sharing with D3D11 or OpenGL
	@param pDevice for D3D11 */
	OvrvisionProOpenCL(int width, int height, enum SHARING_MODE mode = NONE, void *pDevice = NULL);
	~OvrvisionProOpenCL();

	/*! @brief release resources */
	void Close();

	// Select GPU device
	cl_device_id SelectGPU(const char *platform, const char *version);



	//void InspectTextures(uchar* left, uchar *right, uint type = 0);
	static bool CheckGPU();



	// Enumerate OpenCL extensions
	int DeviceExtensions(EXTENSION_CALLBACK callback = NULL, void *item = NULL);

protected:
	/*! @brief Create context with sharing with D3D!! or OpenGL
	@param mode shareing
	@param pDevice to share texture */
	void CreateContext(SHARING_MODE mode, void *pDevice);

	// OpenGL shared textrue
	// pixelFormat must be GL_RGBA
	// dataType must be GL_UNSIGNED_BYTE
	//cl_mem CreateGLTexture2D(GLuint texture, int width, int height);


	void CopyImage(cl_mem left, cl_mem right, cl_event *event_l, cl_event *event_r);
private:
	/*! @brief Download from GPU
	@param image
	@param ptr for read buffer
	@param offsetX
	@param offsetY
	@param width
	@param height */

	bool CreateProgram();
	bool Prepare4Sharing();		// Prepare for OpenGL/D3D sharing
	void createProgram(const char *filename, bool binary = false);
	int saveBinary(const char *filename);
	//bool SaveSettings(const char *filename);


	//clGetGLContextInfoKHR_fn			pclGetGLContextInfoKHR = NULL;

	char	*_deviceExtensions;
	Mat		*_mapX[2], *_mapY[2];	// camera parameter
	Mat		*_skinmask[2];				// skin mask
	Mat		*_histgram[2];
	int		_skinThreshold;
	uint	_width, _height;
	// HSV color region 
	int		_h_low, _h_high;
	int		_s_low, _s_high;
	bool	_remapAvailable;
	bool	_released;
	bool	_calibration;
	int		_frameCounter;
	enum SHARING_MODE _sharing;	// Sharing with OpenGL or Direct3D11 
	enum SCALING	_scaling;	//
	size_t	_scaledRegion[3];
	//Convex	_convex[2];			// Assume to be both hands
	//KalmanFilter _kalman[2];

protected:
	// OpenCL variables
	cl_platform_id	_platformId;
	cl_device_id	_deviceId;
	cl_context		_context;

	cl_command_queue _commandQueue;
	cl_int			_errorCode;

	cl_program		_program;
	cl_kernel		_demosaic;
	cl_kernel		_remap;
	cl_kernel		_resize;
	cl_kernel		_convertHSV;
	cl_kernel		_convertGrayscale;
	cl_kernel		_skincolor;
	cl_kernel		_gaussianBlur3x3;
	cl_kernel		_gaussianBlur5x5;
	cl_kernel		_medianBlur3x3;
	cl_kernel		_medianBlur5x5;
	cl_kernel		_mask;
	cl_kernel		_maskOpengl;
	cl_kernel		_maskD3D11;
	cl_kernel		_invertMask;
	cl_kernel		_copyOpengl;
	// kernels with tone correction
	cl_kernel		_toneCorrection;
	cl_kernel		_resizeTone;
	cl_kernel		_convertHSVTone;
	cl_kernel		_maskTone;
	cl_kernel		_maskOpenglTone;

private:
	cl_mem	_src;
	cl_mem	_l, _r;			// demosaic and remapped image
	cl_mem	_L, _R;			// work image
	cl_mem	_mx[2], _my[2]; // map for remap in GPU
	cl_mem	_reducedL, _reducedR;	// reduced image
	cl_mem	_texture[2];	// Texture sharing
	cl_mem	_toneMap;
	cl_image_desc _desc_scaled;
};