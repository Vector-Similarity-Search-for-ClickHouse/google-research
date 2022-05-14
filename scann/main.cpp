#include "tensorflow/core/platform/status.h"
#include "absl/strings/str_format.h"
#include "scann/utils/common.h"

using tensorflow::Status;

int main() {

    research_scann::InvalidArgumentError(
          absl::StrFormat("datapoint_to_token length=%d but expected %d",
                          10, 10));
    
    std::cout << 123;
}