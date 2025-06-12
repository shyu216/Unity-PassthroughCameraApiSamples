// Copyright (c) Meta Platforms, Inc. and affiliates.

using System.Collections;
using Meta.XR.Samples;
using UnityEngine;
using UnityEngine.UI;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace PassthroughCameraSamples.CameraViewer
{
    [MetaCodeSample("PassthroughCameraApiSamples-EVMViewer")]
    public class EVMViewerManager : MonoBehaviour
    {
        // Create a field to attach the reference to the WebCamTextureManager prefab
        [SerializeField] private WebCamTextureManager m_webCamTextureManager;
        [SerializeField] private Text m_debugText;
        [SerializeField] private RawImage m_image;

        private EvmMagnifier m_yMagnifier;
        private EvmMagnifier m_crMagnifier;
        private EvmMagnifier m_cbMagnifier;
        private Texture2D m_processedTexture;

        private IEnumerator Start()
        {
            while (m_webCamTextureManager.WebCamTexture == null)
            {
                yield return null;
            }

            m_debugText.text += "\nWebCamTexture Object ready and playing.";
            // Set WebCamTexture GPU texture to the RawImage Ui element
            m_image.texture = m_webCamTextureManager.WebCamTexture;

            // Initialize EvmMagnifiers for Y, Cr, and Cb channels
            m_yMagnifier = new EvmMagnifier(alpha: 50, fl: 60 / 60.0, fh: 100 / 60.0, nLevels: 4, fps: 30, attenuation: 1);
            m_crMagnifier = new EvmMagnifier(alpha: 50, fl: 60 / 60.0, fh: 100 / 60.0, nLevels: 4, fps: 30, attenuation: 1);
            m_cbMagnifier = new EvmMagnifier(alpha: 50, fl: 60 / 60.0, fh: 100 / 60.0, nLevels: 4, fps: 30, attenuation: 1);

            Debug.Log("***************************************************");
            Debug.Log("EVMViewerManager initialized with EvmMagnifiers for Y, Cr, and Cb channels.");
            Debug.Log("***************************************************");


            // Initialize processed texture
            m_processedTexture = new Texture2D(
                m_webCamTextureManager.WebCamTexture.width,
                m_webCamTextureManager.WebCamTexture.height,
                TextureFormat.RGBA32,
                false
            );
        }

        private void Update()
        {

            ProcessFrame();

            //m_debugText.text = PassthroughCameraPermissions.HasCameraPermission == true ? "Permission granted." : "No permission granted.";
        }


        private void ProcessFrame()
        {
            var webCamTexture = m_webCamTextureManager.WebCamTexture;
            if (webCamTexture != null)
            {
                // Get the current frame's pixel data
                var pixels = webCamTexture.GetPixels32();

                // Convert to Emgu CV Image format
                var frameImage = new Image<Bgr, byte>(webCamTexture.width, webCamTexture.height);
                for (var y = 0; y < webCamTexture.height; y++)
                {
                    for (var x = 0; x < webCamTexture.width; x++)
                    {
                        var index = y * webCamTexture.width + x;
                        var color = pixels[index];
                        frameImage.Data[y, x, 0] = color.b;
                        frameImage.Data[y, x, 1] = color.g;
                        frameImage.Data[y, x, 2] = color.r;
                    }
                }

                // Convert to YCrCb format
                var yccFrame = frameImage.Convert<Ycc, byte>();
                var yChannel = yccFrame.Split()[0];
                var crChannel = yccFrame.Split()[1];
                var cbChannel = yccFrame.Split()[2];

                // Apply EvmMagnifier to each channel
                var processedY = m_yMagnifier.ProcessFrame(yChannel);
                var processedCr = m_crMagnifier.ProcessFrame(crChannel);
                var processedCb = m_cbMagnifier.ProcessFrame(cbChannel);

                // Reconstruct the YCrCb frame
                var reconstructedYcc = new Image<Ycc, byte>(webCamTexture.width, webCamTexture.height);
                reconstructedYcc[0] = processedY;
                reconstructedYcc[1] = processedCr;
                reconstructedYcc[2] = processedCb;

                // Convert back to BGR format
                var reconstructedBgr = reconstructedYcc.Convert<Bgr, byte>();

                // Convert back to Unity Texture2D
                for (var y = 0; y < webCamTexture.height; y++)
                {
                    for (var x = 0; x < webCamTexture.width; x++)
                    {
                        var index = y * webCamTexture.width + x;
                        var color = reconstructedBgr.Data[y, x, 0];
                        pixels[index] = new Color32(
                            reconstructedBgr.Data[y, x, 2],
                            reconstructedBgr.Data[y, x, 1],
                            reconstructedBgr.Data[y, x, 0],
                            255
                        );
                    }
                }

                m_processedTexture.SetPixels32(pixels);
                m_processedTexture.Apply();

                // Update RawImage texture
                m_image.texture = m_processedTexture;

                m_debugText.text = $"Processed Frame: {webCamTexture.width}x{webCamTexture.height} at {webCamTexture.requestedFPS} FPS";
            }
            else
            {
                Debug.LogWarning("WebCamTexture is not ready.");
            }
        }
    }

}