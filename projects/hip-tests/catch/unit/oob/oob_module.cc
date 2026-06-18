#include <hip_test_common.hh>
#include <string_view>

#define DECL_ELF_FOR_TEST(input_name) constexpr std::string_view input_name = #input_name ".co"

TEST_CASE("OOB_hip_module_load_over") {
  DECL_ELF_FOR_TEST(elf_valid);
  DECL_ELF_FOR_TEST(elf_huge_shnum);
  DECL_ELF_FOR_TEST(elf_bad_shoff);
  DECL_ELF_FOR_TEST(elf_table_spill);
  DECL_ELF_FOR_TEST(elf_sh_overflow);

  SECTION("valid - sanity") {
    hipModule_t module{};
    HIP_CHECK(hipModuleLoad(&module, elf_valid.data()));
    HIP_CHECK(hipModuleUnload(module));
  }

  SECTION("huge shnum") {
    hipModule_t module{};
    HIP_CHECK_ERROR(hipModuleLoad(&module, elf_huge_shnum.data()), hipErrorInvalidImage);
  }

  SECTION("bad shoff") {
    hipModule_t module{};
    HIP_CHECK_ERROR(hipModuleLoad(&module, elf_bad_shoff.data()), hipErrorInvalidImage);
  }

  SECTION("table spill") {
    hipModule_t module{};
    HIP_CHECK_ERROR(hipModuleLoad(&module, elf_table_spill.data()), hipErrorInvalidImage);
  }

  SECTION("sh overflow") {
    hipModule_t module{};
    HIP_CHECK_ERROR(hipModuleLoad(&module, elf_sh_overflow.data()), hipErrorInvalidImage);
  }
}
