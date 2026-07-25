package com.recon3d.capture

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class SessionPhotoStoreTest {
    @get:Rule val tmp = TemporaryFolder()

    @Test fun `newSession creates dir named by clock`() {
        val store = SessionPhotoStore(tmp.root, clock = { 12345L })
        val dir = store.newSession()
        assertTrue(dir.isDirectory)
        assertEquals("12345", dir.name)
    }

    @Test fun `photos returns only jpg sorted by name`() {
        val store = SessionPhotoStore(tmp.root, clock = { 1L })
        val dir = store.newSession()
        File(dir, "b.jpg").writeBytes(byteArrayOf(1))
        File(dir, "a.jpg").writeBytes(byteArrayOf(1))
        File(dir, "note.txt").writeBytes(byteArrayOf(1))
        assertEquals(listOf("a.jpg", "b.jpg"), store.photos(dir).map { it.name })
        assertEquals(2, store.count(dir))
    }

    @Test fun `delete removes session dir`() {
        val store = SessionPhotoStore(tmp.root, clock = { 1L })
        val dir = store.newSession()
        store.delete(dir)
        assertFalse(dir.exists())
    }
}
