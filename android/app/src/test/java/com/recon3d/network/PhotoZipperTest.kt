package com.recon3d.network

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.util.zip.ZipFile

class PhotoZipperTest {
    @get:Rule val tmp = TemporaryFolder()

    @Test fun `zip packs all jpgs flat and returns count`() {
        val dir = tmp.newFolder("photos")
        File(dir, "1.jpg").writeBytes(byteArrayOf(10, 20))
        File(dir, "2.jpg").writeBytes(byteArrayOf(30))
        File(dir, "ignore.txt").writeBytes(byteArrayOf(0))
        val dest = File(tmp.root, "out.zip")

        val count = PhotoZipper.zip(dir, dest)

        assertEquals(2, count)
        ZipFile(dest).use { zf ->
            val names = zf.entries().toList().map { it.name }.sorted()
            assertEquals(listOf("1.jpg", "2.jpg"), names)
            assertEquals(2, zf.getEntry("1.jpg").size)
        }
    }

    @Test(expected = IllegalArgumentException::class)
    fun `empty dir throws`() {
        PhotoZipper.zip(tmp.newFolder("empty"), File(tmp.root, "x.zip"))
    }
}
