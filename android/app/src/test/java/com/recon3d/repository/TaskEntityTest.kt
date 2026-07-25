package com.recon3d.repository

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class TaskEntityTest {
    private fun entity(state: String) = TaskEntity("id", 0L, state)

    @Test fun `active states`() {
        assertTrue(entity("training").isActive())
        assertTrue(entity("uploaded").isActive())
        assertFalse(entity("done").isActive())
        assertFalse(entity("failed").isActive())
    }

    @Test fun `state labels`() {
        assertEquals("3DGS 训练", stateLabel("training"))
        assertEquals("完成", stateLabel("done"))
        assertEquals("失败", stateLabel("failed"))
        assertEquals("whatever", stateLabel("whatever"))
    }
}
