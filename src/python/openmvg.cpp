#include <pybind11/pybind11.h>

#include "openMVG/image/image.hpp"
#include "openMVG/sfm/sfm.hpp"

/// Feature/Regions & Image describer interfaces
#include "openMVG/features/features.hpp"
#include "openMVG/features/sift/SIFT_Anatomy_Image_Describer.hpp"
#include "nonFree/sift/SIFT_describer.hpp"
#include <cereal/archives/json.hpp>
#include "openMVG/system/timer.hpp"

#include "third_party/cmdLine/cmdLine.h"
#include "third_party/stlplus3/filesystemSimplified/file_system.hpp"
#include "third_party/progress/progress.hpp"

#include <cstdlib>
#include <fstream>

#ifdef OPENMVG_USE_OPENMP
#include <omp.h>
#endif

using namespace openMVG;
using namespace openMVG::image;
using namespace openMVG::features;
using namespace openMVG::sfm;
using namespace std;

namespace py = pybind11;

bool run(std::string filename, std::string outdir) {
  //---------------------------------------
  // a. Load input scene
  //---------------------------------------
  SfM_Data sfm_data;
  if (!Load(sfm_data, filename, ESfM_Data(VIEWS|INTRINSICS))) {
    std::cerr << std::endl
      << "The input file \""<< filename << "\" cannot be read" << std::endl;
    return false;
  }

  // b. Init the image_describer
  // - retrieve the used one in case of pre-computed features
  // - else create the desired one

  using namespace openMVG::features;
  std::unique_ptr<Image_describer> image_describer;

  const std::string sImage_describer = stlplus::create_filespec(outdir, "image_describer", "json");
  // Create the desired Image_describer method.
  // Don't use a factory, perform direct allocation
  image_describer.reset(new SIFT_Image_describer
    (SIFT_Image_describer::Params(), true));
  if (!image_describer->Set_configuration_preset(features::ULTRA_PRESET))
  {
    std::cerr << "Preset configuration failed." << std::endl;
    return EXIT_FAILURE;
  }

  std::ofstream stream(sImage_describer.c_str());
  if (!stream.is_open())
    return false;

  cereal::JSONOutputArchive archive(stream);
  archive(cereal::make_nvp("image_describer", image_describer));
  std::unique_ptr<Regions> regionsType;
  image_describer->Allocate(regionsType);
  archive(cereal::make_nvp("regions_type", regionsType));

  // Feature extraction routines
  // For each View of the SfM_Data container:
  // - if regions file exists continue,
  // - if no file, compute features
  Image<unsigned char> imageGray, globalMask;

  const std::string sGlobalMask_filename = stlplus::create_filespec(outdir, "mask.png");
  if (stlplus::file_exists(sGlobalMask_filename))
  {
    if (ReadImage(sGlobalMask_filename.c_str(), &globalMask))
    {
      std::cout
        << "Feature extraction will use a GLOBAL MASK:\n"
        << sGlobalMask_filename << std::endl;
    }
  }

  const unsigned int nb_max_thread = omp_get_max_threads();

  omp_set_num_threads(nb_max_thread);

  for(int i = 0; i < static_cast<int>(sfm_data.views.size()); ++i)
  {
    Views::const_iterator iterViews = sfm_data.views.begin();
    std::advance(iterViews, i);
    const View * view = iterViews->second.get();
    const std::string
      sView_filename = stlplus::create_filespec(sfm_data.s_root_path, view->s_Img_path),
      sFeat = stlplus::create_filespec(outdir, stlplus::basename_part(sView_filename), "feat"),
      sDesc = stlplus::create_filespec(outdir, stlplus::basename_part(sView_filename), "desc");

    if (!ReadImage(sView_filename.c_str(), &imageGray))
      continue;

    Image<unsigned char> * mask = nullptr; // The mask is null by default

    const std::string sImageMask_filename =
      stlplus::create_filespec(sfm_data.s_root_path,
        stlplus::basename_part(sView_filename) + "_mask", "png");

    Image<unsigned char> imageMask;
    if (stlplus::file_exists(sImageMask_filename))
      ReadImage(sImageMask_filename.c_str(), &imageMask);

    // The mask point to the globalMask, if a valid one exists for the current image
    if (globalMask.Width() == imageGray.Width() && globalMask.Height() == imageGray.Height())
      mask = &globalMask;
    // The mask point to the imageMask (individual mask) if a valid one exists for the current image
    if (imageMask.Width() == imageGray.Width() && imageMask.Height() == imageGray.Height())
      mask = &imageMask;

    // Compute features and descriptors and export them to files
    std::unique_ptr<Regions> regions;
    image_describer->Describe(imageGray, regions, mask);
    image_describer->Save(regions.get(), sFeat, sDesc);
  }
  return EXIT_SUCCESS;
}

bool process_image(View view, std::string image_path, std::string mask_path, std::string infofile, std::string descfile, std::string featfile)
{
  Image<unsigned char> image;

  if (!ReadImage(image_path.c_str(), &image))
    return false;

  std::unique_ptr<Image_describer> image_describer;

  image_describer.reset(new SIFT_Image_describer
    (SIFT_Image_describer::Params(), true));
  if (!image_describer->Set_configuration_preset(features::ULTRA_PRESET))
  {
    std::cerr << "Preset configuration failed." << std::endl;
    return EXIT_FAILURE;
  }

  std::ofstream stream(infofile.c_str());
  if (!stream.is_open())
    return false;

  cereal::JSONOutputArchive archive(stream);
  archive(cereal::make_nvp("image_describer", image_describer));
  std::unique_ptr<Regions> regionsType;
  image_describer->Allocate(regionsType);
  archive(cereal::make_nvp("regions_type", regionsType));

  Image<unsigned char> * mask = nullptr;
  if (stlplus::file_exists(mask_path))
    ReadImage(mask_path.c_str(), mask);

  // Compute features and descriptors and export them to files
  std::unique_ptr<Regions> regions;
  image_describer->Describe(image, regions, mask);
  image_describer->Save(regions.get(), featfile, descfile);
}

PYBIND11_PLUGIN(openmvg) {
    py::module m("openmvg", "openMVG Bindings");

    m.def("run", &run, "Compute image features");
    m.def("process_image", &process_image, "Generate features and sift descriptors for a single image");

    py::class_<View>(m, "View")
        .def(py::init<const std::string &, IndexT, IndexT, IndexT, IndexT, IndexT>());

    return m.ptr();
}

