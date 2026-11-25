import groovy.json.JsonSlurper
import java.security.MessageDigest

plugins {
    alias(libs.plugins.android.library)
}

val kitPath = providers.gradleProperty("qoldaEngineKit").orNull.orEmpty()
val sdkPath = providers.gradleProperty("qoldaLiteRtSdk").orNull.orEmpty()
val kitDir = file(kitPath.ifEmpty { "missing-kit" })
val sdkDir = file(sdkPath.ifEmpty { "missing-sdk" })

fun sha256(file: File): String {
    val digest = MessageDigest.getInstance("SHA-256")
    file.inputStream().use { stream ->
        val buffer = ByteArray(1 shl 20)
        while (true) {
            val read = stream.read(buffer)
            if (read <= 0) break
            digest.update(buffer, 0, read)
        }
    }
    return digest.digest().joinToString("") { "%02x".format(it) }
}

// The kit is the byte-verified payload from T-017. Refuse to build against anything else.
val verifyEngineKit by tasks.registering {
    val kit = kitDir
    val sdk = sdkDir
    inputs.dir(kit.resolve("jniLibs"))
    inputs.dir(kit.resolve("assets/qolda"))
    doLast {
        if (kitPath.isEmpty() || sdkPath.isEmpty()) {
            throw GradleException("pass -PqoldaEngineKit=<m4-2-engine-input-v1> and -PqoldaLiteRtSdk=<litert-2.4.0>")
        }
        val manifest = JsonSlurper().parse(kit.resolve("assets/qolda/engine.json")) as Map<*, *>
        val graphs = manifest["graphs"] as Map<*, *>
        for ((name, raw) in graphs) {
            val graph = raw as Map<*, *>
            val asset = kit.resolve(graph["path"] as String)
            if (!asset.isFile) throw GradleException("$name: missing ${asset.name}")
            if (asset.length() != (graph["bytes"] as Number).toLong()) throw GradleException("$name: size mismatch")
            val digest = sha256(asset)
            if (digest != graph["sha256"]) throw GradleException("$name: sha256 mismatch ($digest)")
            if (graph["dtype"] != "float32") throw GradleException("$name: expected FP32")
            if (graph["signature"] != "serving_default") throw GradleException("$name: unexpected signature")
        }
        val kitLibs = kit.resolve("jniLibs/arm64-v8a")
        val sdkLibs = sdk.resolve("jni/arm64-v8a")
        val local = kitLibs.resolve("libLiteRt.so")
        if (!local.isFile) throw GradleException("kit is missing libLiteRt.so")
        val sdkLib = sdkLibs.resolve("libLiteRt.so")
        if (sdkLib.isFile && sha256(sdkLib) != sha256(local)) {
            throw GradleException("libLiteRt.so differs between the kit and the LiteRT SDK")
        }
        // The accelerator is packaged from the kit: those exact bytes were delegated on the phone.
        // A same-size rebuild of the delegate in the SDK is not a failure, but it is reported.
        val kitDelegate = kitLibs.resolve("libLiteRtClGlAccelerator.so")
        if (!kitDelegate.isFile) throw GradleException("kit is missing libLiteRtClGlAccelerator.so")
        val sdkDelegate = sdkLibs.resolve("libLiteRtClGlAccelerator.so")
        if (sdkDelegate.isFile && sha256(sdkDelegate) != sha256(kitDelegate)) {
            logger.lifecycle("note: libLiteRtClGlAccelerator.so in the SDK differs from the verified kit bytes; the kit copy is packaged")
        }
        if (!sdk.resolve("sdk/litert_cc_sdk/litert/c/litert_common.h").isFile) {
            throw GradleException("LiteRT SDK headers are missing under $sdk")
        }
    }
}

android {
    namespace = "net.qolda.engine"
    compileSdk = 34
    ndkVersion = "26.1.10909125"

    defaultConfig {
        minSdk = 26
        consumerProguardFiles("consumer-rules.pro")
        ndk { abiFilters += listOf("arm64-v8a") }
        externalNativeBuild {
            cmake {
                // Static libc++ keeps the runtime set to exactly the verified LiteRT bytes.
                arguments += listOf(
                    "-DANDROID_STL=c++_static",
                    "-DQOLDA_LITERT_SDK=${sdkDir.absolutePath}",
                )
                cppFlags += "-std=c++17"
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    sourceSets["main"].jniLibs.srcDirs(kitDir.resolve("jniLibs"))

    packaging {
        jniLibs {
            // The engine links libc++ statically and LiteRT needs no C++ runtime.
            excludes += "**/libc++_shared.so"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    lint { abortOnError = false }

    tasks.withType<com.android.build.gradle.tasks.ExternalNativeBuildTask>().configureEach {
        dependsOn(verifyEngineKit)
    }
    tasks.matching { it.name == "preBuild" }.configureEach { dependsOn(verifyEngineKit) }
}

dependencies {
}
