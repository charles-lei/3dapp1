package com.recon3d.network

import java.io.File
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

object PhotoZipper {
    /** 把 photoDir 内全部 .jpg 平铺打包到 destZip，返回张数。 */
    fun zip(photoDir: File, destZip: File): Int {
        val photos = photoDir.listFiles { f -> f.extension.lowercase() == "jpg" }?.sortedBy { it.name }
            ?: emptyList()
        require(photos.isNotEmpty()) { "没有可上传的照片" }
        destZip.parentFile?.mkdirs()
        ZipOutputStream(destZip.outputStream().buffered()).use { zos ->
            for (photo in photos) {
                zos.putNextEntry(ZipEntry(photo.name))
                photo.inputStream().buffered().use { it.copyTo(zos) }
                zos.closeEntry()
            }
        }
        return photos.size
    }
}
