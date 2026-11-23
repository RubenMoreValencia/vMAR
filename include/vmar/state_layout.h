#ifndef VMAR_STATE_LAYOUT_H
#define VMAR_STATE_LAYOUT_H

#include "vmar/vmar.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Pure structural description. It does not select a field, integration
 * method, controller or learning algorithm. All sizes are in doubles. */
typedef enum vmar_state_kind {
    VMAR_STATE_PARAMETERS = 1,
    VMAR_STATE_FIRST_MEMORY = 2,
    VMAR_STATE_SECOND_MOMENT = 3
} vmar_state_kind;

typedef struct vmar_state_layout {
    vmar_state_kind kind;
    size_t parameters;
    size_t theta_offset;
    size_t first_offset;
    size_t second_offset;
    size_t total_doubles;
} vmar_state_layout;

/* Offsets into contiguous caller-owned memory. For absent blocks, offsets
 * equal total_doubles. The second moment is only a layout capability:
 * its candidate dynamics remain outside vmar_core. */
vmar_status vmar_state_layout_make(size_t parameters, vmar_state_kind kind,
                                   vmar_state_layout *result);

/* Check that layout is canonical, values are finite and (if present)
 * the second moment is nonnegative. No mutation or hidden allocation. */
vmar_status vmar_state_layout_validate(const vmar_state_layout *layout,
                                       const double *state,
                                       size_t supplied_doubles);

#ifdef __cplusplus
}
#endif
#endif
