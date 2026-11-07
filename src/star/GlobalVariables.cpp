#include "GlobalVariables.h"
Stats g_statsAll;//global mapping statistics
ThreadControl g_threadChunks;

#include "star_api.h"
StarReadSource* g_starReadSource=nullptr;//singlet: in-process read source (star_api.h)
