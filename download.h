/* SPDX-License-Identifier: Unlicense */

#ifndef I_SAM3_OPENCL_DOWNLOAD_H
#define I_SAM3_OPENCL_DOWNLOAD_H

/* Download url to path unless path already exists with size bytes
 * (size 0: any size). The file is written to path.part (resumed if an
 * earlier download was interrupted), checked against size and renamed
 * into place; concurrent processes fetching the same file wait for each
 * other. Missing parent folders are created. header, if not NULL, is
 * an extra HTTP request header (e.g. authorization). Fatal on any
 * failure. */
void download_file(const char *url, const char *path, long long size,
                   const char *header);

#endif /* I_SAM3_OPENCL_DOWNLOAD_H */
